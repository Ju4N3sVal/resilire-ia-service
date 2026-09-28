"""
RESILIRE — Fase IA Sprint 3
Entrenamiento reproducible del modelo de segmentación territorial (K-means)
según la configuración de referencia del documento PI2-RES-S2-04:

  unidad: municipio (87 de Santander) · variables: 15 privaciones estandarizadas
  (se excluyen código e IPM como predictores) · k en 2..5 · semilla 42 · n_init 20
  métricas: silhouette e inercia (+ Davies-Bouldin, Calinski-Harabasz y estabilidad)

Además genera:
  - correlaciones descriptivas de cada privación con el IPM (no causales),
  - impacto de variables sobre la segmentación (ANOVA F, importancia por
    permutación de un modelo sustituto),
  - métricas de precisión del modelo sustituto (árbol de decisión que
    reproduce los grupos con validación cruzada estratificada),
  - los artefactos listados en la sección 8 del documento del Sprint 2.

Uso:
  python -m training.train --input data/raw/anexo-censal-pobreza-municipal-2018.xlsx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .dane_loader import FEATURES, PRIVACIONES, load_anexo

SEED = 42
N_INIT = 20
K_RANGE = range(2, 6)
DEPTO_SANTANDER = "68"
BUCARAMANGA = "68001"
MODEL_VERSION = "baseline_kmeans_v1"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reglas de limpieza 1-6 del documento del Sprint 2."""
    df = df.copy()
    df["codigo"] = df["codigo"].astype(str).str.zfill(5)
    for c in ["ipm", *FEATURES]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Algunas publicaciones expresan proporciones 0-1: se normaliza a 0-100
    for c in ["ipm", *FEATURES]:
        if df[c].max(skipna=True) is not None and df[c].max(skipna=True) <= 1.0:
            df[c] = df[c] * 100
    sant = df[df["codigo"].str.startswith(DEPTO_SANTANDER)].copy()

    rows = []
    for c in ["ipm", *FEATURES]:
        s = sant[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = sant.loc[(s < lo) | (s > hi), "municipio"].astype(str).tolist() if "municipio" in sant else []
        rows.append({
            "variable": c,
            "dimension": PRIVACIONES[c][0] if c in PRIVACIONES else "Índice",
            "n": int(s.notna().sum()),
            "faltantes": int(s.isna().sum()),
            "fuera_rango_0_100": int(((s < 0) | (s > 100)).sum()),
            "media": round(float(s.mean()), 2),
            "desv_est": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "atipicos_iqr": len(outliers),
            "municipios_atipicos": "; ".join(outliers),
            "tratamiento": "Atípicos conservados y marcados (diferencias territoriales reales)",
        })
    quality = pd.DataFrame(rows)
    dups = int(sant["codigo"].duplicated().sum())
    quality.attrs["duplicados_codigo"] = dups
    return sant.reset_index(drop=True), quality


def evaluate_k(X: np.ndarray) -> tuple[list[dict], dict]:
    results, models = [], {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=N_INIT)
        labels = km.fit_predict(X)
        results.append({
            "k": k,
            "silhouette": round(float(silhouette_score(X, labels)), 4),
            "inercia": round(float(km.inertia_), 2),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 2),
            "tamanos": sorted([int(v) for v in np.bincount(labels)], reverse=True),
        })
        models[k] = km
    return results, models


def stability(X: np.ndarray, k: int, reference: np.ndarray, n_boot: int = 100) -> dict:
    rng = np.random.default_rng(SEED)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        km = KMeans(n_clusters=k, random_state=SEED + b + 1, n_init=N_INIT).fit(X[idx])
        aris.append(adjusted_rand_score(reference, km.predict(X)))
    aris = np.array(aris)
    return {"metodo": f"Bootstrap ({n_boot} remuestreos) · ARI frente a la partición de referencia",
            "ari_media": round(float(aris.mean()), 4),
            "ari_p05": round(float(np.percentile(aris, 5)), 4),
            "ari_p95": round(float(np.percentile(aris, 95)), 4)}


def main(input_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    raw = load_anexo(input_path)
    df, quality = clean(raw)

    required = ["ipm", *FEATURES]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Faltan columnas tras la lectura del anexo: {missing_cols}")
    if df[FEATURES].isna().any().any():
        raise SystemExit("Hay faltantes en las privaciones: revisar antes de entrenar (no se imputa sin justificación).")

    X_raw = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    # 1. Selección de k
    k_results, models = evaluate_k(X)
    best = max(k_results, key=lambda r: r["silhouette"])
    k = best["k"]
    km = models[k]
    labels = km.labels_

    # Reordenar clusters por IPM medio ascendente (0 = menor privación) para lectura estable
    order = df.assign(c=labels).groupby("c")["ipm"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    km.cluster_centers_ = km.cluster_centers_[order]
    km.labels_ = labels
    df["cluster_id"] = labels
    df["silhouette_municipio"] = np.round(silhouette_samples(X, labels), 4)

    # 2. Estabilidad
    stab = stability(X, k, labels)

    # 3. Perfiles de grupo (IPM solo para interpretación ex post, no es insumo)
    profiles = []
    sant_mean = df[FEATURES].mean()
    for c in range(k):
        sub = df[df["cluster_id"] == c]
        means = sub[FEATURES].mean()
        z = (means - sant_mean) / df[FEATURES].std()
        rasgos = z.abs().sort_values(ascending=False).head(4).index.tolist()
        profiles.append({
            "cluster_id": c,
            "n_municipios": int(len(sub)),
            "ipm_medio_interpretativo": round(float(sub["ipm"].mean()), 2),
            "ipm_rango": [round(float(sub["ipm"].min()), 2), round(float(sub["ipm"].max()), 2)],
            "privaciones_media": {f: round(float(means[f]), 2) for f in FEATURES},
            "rasgos_distintivos": [{"variable": f, "media_grupo": round(float(means[f]), 2),
                                    "media_santander": round(float(sant_mean[f]), 2),
                                    "z": round(float(z[f]), 2)} for f in rasgos],
            "municipios": sub.sort_values("ipm")["municipio"].astype(str).tolist() if "municipio" in sub else [],
        })

    # 4. Impacto de variables
    F, p = f_classif(X, labels)
    surrogate = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    n_splits = int(min(5, np.bincount(labels).min()))
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=SEED)
    cvres = cross_validate(surrogate, X, labels, cv=cv,
                           scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
    surrogate.fit(X, labels)
    perm = permutation_importance(surrogate, X, labels, n_repeats=50, random_state=SEED)
    importance = sorted([
        {"variable": f, "dimension": PRIVACIONES[f][0],
         "anova_f": round(float(F[i]), 2), "p_valor": float(f"{p[i]:.3g}"),
         "importancia_permutacion": round(float(perm.importances_mean[i]), 4),
         "importancia_arbol": round(float(surrogate.feature_importances_[i]), 4)}
        for i, f in enumerate(FEATURES)], key=lambda r: r["anova_f"], reverse=True)

    # 5. Correlaciones descriptivas con el IPM (Santander)
    corr = sorted([
        {"variable": f, "dimension": PRIVACIONES[f][0],
         "pearson": round(float(df[f].corr(df["ipm"])), 2),
         "spearman": round(float(df[f].corr(df["ipm"], method="spearman")), 2)}
        for f in FEATURES], key=lambda r: r["pearson"], reverse=True)

    # 6. Artefactos
    pipe = Pipeline([("scaler", scaler), ("kmeans", km)])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "model_version_id": MODEL_VERSION,
        "tipo": "Segmentación no supervisada (K-means) — no es predicción del IPM",
        "entrenado_utc": now,
        "sklearn": sklearn.__version__,
        "fuente": "DANE · Medida de pobreza multidimensional municipal de fuente censal (CNPV 2018)",
        "archivo_fuente": os.path.basename(input_path),
        "sha256_fuente": sha256(input_path),
        "unidad": "municipio", "universo": "Santander", "n_municipios": int(len(df)),
        "features": FEATURES, "k": k, "semilla": SEED, "n_init": N_INIT,
        "criterio_k": "Máximo silhouette en k=2..5",
        "clusters_ordenados_por": "IPM medio ascendente (0 = menor incidencia)",
    }
    joblib.dump({"pipeline": pipe, "meta": meta, "profiles": profiles}, os.path.join(out_dir, "baseline_model_v1.joblib"))

    metrics = {
        **meta,
        "seleccion_k": k_results,
        "k_seleccionado": best,
        "estabilidad": stab,
        "modelo_sustituto": {
            "descripcion": "Árbol de decisión (profundidad 3) que reproduce los grupos; mide qué tan explicable es la segmentación",
            "validacion": f"StratifiedKFold({cv.n_splits}), semilla {SEED}",
            "accuracy": round(float(cvres["test_accuracy"].mean()), 4),
            "accuracy_desv": round(float(cvres["test_accuracy"].std()), 4),
            "precision_macro": round(float(cvres["test_precision_macro"].mean()), 4),
            "recall_macro": round(float(cvres["test_recall_macro"].mean()), 4),
            "f1_macro": round(float(cvres["test_f1_macro"].mean()), 4),
        },
        "calidad": {"duplicados_codigo": quality.attrs["duplicados_codigo"],
                    "faltantes_total": int(quality["faltantes"].sum()),
                    "fuera_rango_total": int(quality["fuera_rango_0_100"].sum())},
    }
    with open(os.path.join(out_dir, "metrics_baseline_v1.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "profiles_v1.json"), "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "importance_v1.json"), "w", encoding="utf-8") as f:
        json.dump(importance, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "correlations_v1.json"), "w", encoding="utf-8") as f:
        json.dump(corr, f, ensure_ascii=False, indent=2)
    schema = {"version": "input_schema_v1", "tipo": "object",
              "campos": {f: {"tipo": "number", "min": 0, "max": 100, "unidad": "% de hogares",
                             "dimension": PRIVACIONES[f][0]} for f in FEATURES},
              "requeridos": FEATURES}
    with open(os.path.join(out_dir, "input_schema_v1.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    cols = ["codigo", "municipio", "ipm", *FEATURES]
    df[[c for c in cols if c in df.columns] + ["cluster_id", "silhouette_municipio"]] \
        .to_csv(os.path.join(out_dir, "segments_v1.csv"), index=False, encoding="utf-8")
    df[[c for c in cols if c in df.columns]].to_csv(os.path.join(out_dir, "data_processed_v1.csv"), index=False, encoding="utf-8")
    quality.to_csv(os.path.join(out_dir, "quality_report_v1.csv"), index=False, encoding="utf-8")
    pd.DataFrame([{"variable": f, "tipo": "float", "unidad": "% de hogares (0-100)",
                   "dimension": PRIVACIONES[f][0], "fuente": "DANE CNPV 2018",
                   "rol_en_modelo": "Insumo de segmentación (estandarizado)"} for f in FEATURES]
                 + [{"variable": "ipm", "tipo": "float", "unidad": "% de hogares (0-100)", "dimension": "Índice",
                     "fuente": "DANE CNPV 2018", "rol_en_modelo": "Excluido como insumo (matriz de fuga); solo interpretación"},
                    {"variable": "codigo", "tipo": "texto (5)", "unidad": "", "dimension": "Identificador",
                     "fuente": "DANE DIVIPOLA", "rol_en_modelo": "Excluido de X; identificador"}]) \
        .to_csv(os.path.join(out_dir, "data_dictionary_v1.csv"), index=False, encoding="utf-8")

    sur = metrics["modelo_sustituto"]
    card = [
        f"# Model card — {MODEL_VERSION}", "",
        "**Propósito.** Segmentar los 87 municipios de Santander por similitud de su perfil de privaciones "
        "(CNPV 2018) para apoyar la exploración territorial. No predice el IPM, no es causal y no aplica a hogares o personas.", "",
        f"**Datos.** {meta['fuente']}. Archivo `{meta['archivo_fuente']}` (sha256 {meta['sha256_fuente'][:16]}…). "
        f"{len(df)} municipios; {metrics['calidad']['faltantes_total']} faltantes; {metrics['calidad']['duplicados_codigo']} códigos duplicados.", "",
        f"**Configuración.** StandardScaler + K-means; k evaluado de 2 a 5; semilla {SEED}; n_init {N_INIT}; "
        f"k seleccionado = {k} por máximo silhouette. Insumos: 15 privaciones. Excluidos: código municipal e IPM (matriz de fuga).", "",
        "**Métricas.**", "",
        "| k | Silhouette | Inercia | Davies-Bouldin | Calinski-Harabasz | Tamaños |", "|---|---|---|---|---|---|",
        *[f"| {r['k']} | {r['silhouette']} | {r['inercia']} | {r['davies_bouldin']} | {r['calinski_harabasz']} | {r['tamanos']} |" for r in k_results], "",
        f"Estabilidad (bootstrap, ARI): media {stab['ari_media']} (p5 {stab['ari_p05']} – p95 {stab['ari_p95']}).", "",
        f"Modelo sustituto explicativo (árbol, profundidad 3, {sur['validacion']}): accuracy {sur['accuracy']} ± {sur['accuracy_desv']}, "
        f"precisión macro {sur['precision_macro']}, recall macro {sur['recall_macro']}, F1 macro {sur['f1_macro']}.", "",
        "**Limitaciones.** Línea base 2018; unidad municipal (sesgo ecológico si se lee a nivel individual); 87 observaciones; "
        "la separación entre grupos es moderada, por lo que la pertenencia de municipios en el borde entre grupos debe leerse con cautela.", "",
        "**Uso.** Orientativo y no vinculante. La activación del modelo requiere aprobación del Comité IA / Product Owner.",
    ]
    with open(os.path.join(out_dir, "model_card_baseline_v1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(card) + "\n")

    buc = df[df["codigo"] == BUCARAMANGA]
    print(json.dumps({"n": len(df), "k": k, "silhouette": best["silhouette"],
                      "bucaramanga": buc[["ipm", "cluster_id"]].to_dict("records")}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    main(a.input, a.out)
