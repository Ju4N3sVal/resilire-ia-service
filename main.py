"""
RESILIRE — Microservicio de IA (Sprint 5)

Expone el modelo de segmentación territorial versionado (baseline_kmeans_v1)
mediante una API REST /api/v1 documentada con OpenAPI (Swagger en /docs).

Contratos alineados con el documento de arquitectura (sección 6.5):
  GET  /health                       disponibilidad del servicio y del modelo
  GET  /api/v1/model                 ficha del modelo, versión y métricas
  GET  /api/v1/segments              municipios de Santander con su grupo
  GET  /api/v1/segments/{codigo}     grupo y contexto de un municipio
  GET  /api/v1/profiles              perfil promedio de cada grupo
  GET  /api/v1/importance            impacto de variables en la segmentación
  GET  /api/v1/correlations          correlaciones descriptivas con el IPM
  POST /api/v1/predictions           asigna un perfil de privaciones a un grupo
  GET  /api/v1/variables             catálogo de las 15 privaciones (dimensión, etiqueta, estadísticas)
  POST /api/v1/scenarios             compara un escenario base contra uno alternativo (Sprint 5, RF-13)
  GET  /api/v1/priorities            priorización orientativa de municipios (Sprint 5, RF-12)

Toda salida es orientativa y no vinculante (RF-12, RNF-21).
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ART = os.getenv("RESILIRE_ARTIFACTS", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
ORIGINS = [o.strip() for o in os.getenv(
    "RESILIRE_CORS_ORIGINS",
    "https://resilire-web.web.app,https://resilire-web.firebaseapp.com,http://localhost:5173").split(",")]

ADVERTENCIA = ("Resultado orientativo y no vinculante. La segmentación agrupa municipios por similitud de "
               "privaciones (CNPV 2018, línea base); no predice el IPM, no es causal y no aplica a hogares o personas.")

app = FastAPI(title="RESILIRE · Microservicio de IA", version="1.1.0",
              description="Segmentación territorial de vulnerabilidad multidimensional — Santander (CNPV 2018).")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"],
                   expose_headers=["X-Correlation-ID"])

STATE: dict = {}


def _load():
    bundle = joblib.load(os.path.join(ART, "baseline_model_v1.joblib"))
    STATE["pipeline"] = bundle["pipeline"]
    STATE["meta"] = bundle["meta"]
    STATE["profiles"] = bundle["profiles"]
    for name in ("metrics_baseline_v1", "importance_v1", "correlations_v1", "input_schema_v1"):
        with open(os.path.join(ART, f"{name}.json"), encoding="utf-8") as f:
            STATE[name] = json.load(f)
    with open(os.path.join(ART, "segments_v1.csv"), encoding="utf-8") as f:
        STATE["segments"] = list(csv.DictReader(f))


try:
    _load()
except Exception as e:  # el servicio arranca en modo degradado
    STATE["error"] = str(e)


def _cid(request: Request) -> str:
    return getattr(request.state, "correlation_id", "n/a")


@app.middleware("http")
async def correlation(request: Request, call_next):
    request.state.correlation_id = request.headers.get("X-Correlation-ID") or f"corr-{uuid.uuid4().hex[:12]}"
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = request.state.correlation_id
    return response


def error(request: Request, status: int, code: str, message: str, details=None):
    return JSONResponse(status_code=status, content={
        "code": code, "message": message, "correlationId": _cid(request), "details": details or []})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return error(request, 422, "INPUT_VALIDATION_ERROR", "La solicitud contiene valores inválidos.",
                 [{"field": ".".join(str(x) for x in e["loc"][1:]), "issue": e["msg"]} for e in exc.errors()])


def _require_model(request: Request):
    if "pipeline" not in STATE:
        return error(request, 503, "MODEL_UNAVAILABLE", "No hay un modelo activo cargado.",
                     [{"issue": STATE.get("error", "artefactos no encontrados")}])
    return None


@app.get("/health", tags=["Operaciones"])
def health():
    ok = "pipeline" in STATE
    return {"status": "available" if ok else "degraded", "service": "resilire-ia",
            "model_version_id": STATE.get("meta", {}).get("model_version_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@app.get("/api/v1/model", tags=["Modelo"])
def model_card(request: Request):
    if (e := _require_model(request)): return e
    return {"model": STATE["meta"], "metrics": STATE["metrics_baseline_v1"], "advertencia": ADVERTENCIA}


@app.get("/api/v1/segments", tags=["Segmentación"])
def segments(request: Request, cluster_id: int | None = None):
    if (e := _require_model(request)): return e
    rows = STATE["segments"]
    if cluster_id is not None:
        rows = [r for r in rows if int(r["cluster_id"]) == cluster_id]
    return {"model_version_id": STATE["meta"]["model_version_id"], "total": len(rows),
            "items": [{"codigo": r["codigo"], "municipio": r.get("municipio"), "ipm": float(r["ipm"]),
                       "cluster_id": int(r["cluster_id"]), "silhouette": float(r["silhouette_municipio"])} for r in rows],
            "advertencia": ADVERTENCIA}


@app.get("/api/v1/segments/{codigo}", tags=["Segmentación"])
def segment(codigo: str, request: Request):
    if (e := _require_model(request)): return e
    row = next((r for r in STATE["segments"] if r["codigo"] == codigo.zfill(5)), None)
    if row is None:
        return error(request, 404, "TERRITORY_NOT_FOUND", "El código no corresponde a un municipio de Santander.",
                     [{"field": "codigo", "issue": codigo}])
    c = int(row["cluster_id"])
    return {"codigo": row["codigo"], "municipio": row.get("municipio"), "ipm": float(row["ipm"]),
            "cluster_id": c, "silhouette": float(row["silhouette_municipio"]),
            "privaciones": {f: float(row[f]) for f in STATE["meta"]["features"]},
            "perfil": STATE["profiles"][c], "model_version_id": STATE["meta"]["model_version_id"],
            "advertencia": ADVERTENCIA}


@app.get("/api/v1/profiles", tags=["Segmentación"])
def profiles(request: Request):
    if (e := _require_model(request)): return e
    return {"model_version_id": STATE["meta"]["model_version_id"], "items": STATE["profiles"], "advertencia": ADVERTENCIA}


@app.get("/api/v1/importance", tags=["Explicabilidad"])
def importance(request: Request):
    if (e := _require_model(request)): return e
    return {"model_version_id": STATE["meta"]["model_version_id"], "items": STATE["importance_v1"],
            "nota": "Impacto de cada privación en la separación de los grupos (ANOVA F e importancia por permutación)."}


@app.get("/api/v1/correlations", tags=["Explicabilidad"])
def correlations(request: Request):
    if (e := _require_model(request)): return e
    return {"universo": "87 municipios de Santander · CNPV 2018", "items": STATE["correlations_v1"],
            "nota": "Correlaciones descriptivas con el IPM; no implican causalidad."}


class PrivacionesIn(BaseModel):
    analfabetismo: float = Field(ge=0, le=100)
    bajo_logro_educativo: float = Field(ge=0, le=100)
    barreras_primera_infancia: float = Field(ge=0, le=100)
    inasistencia_escolar: float = Field(ge=0, le=100)
    rezago_escolar: float = Field(ge=0, le=100)
    trabajo_infantil: float = Field(ge=0, le=100)
    barreras_acceso_salud: float = Field(ge=0, le=100)
    sin_aseguramiento_salud: float = Field(ge=0, le=100)
    tasa_dependencia: float = Field(ge=0, le=100)
    trabajo_informal: float = Field(ge=0, le=100)
    hacinamiento_critico: float = Field(ge=0, le=100)
    eliminacion_excretas: float = Field(ge=0, le=100)
    paredes_inadecuadas: float = Field(ge=0, le=100)
    pisos_inadecuados: float = Field(ge=0, le=100)
    sin_fuente_agua_mejorada: float = Field(ge=0, le=100)


def _assign(values: list[float]) -> dict:
    """Asigna un vector de 15 privaciones (escala original) al grupo más cercano del modelo activo."""
    pipe = STATE["pipeline"]
    x = np.array([values], dtype=float)
    z = pipe.named_steps["scaler"].transform(x)
    dist = pipe.named_steps["kmeans"].transform(z)[0]
    c = int(np.argmin(dist))
    sd = np.sort(dist)
    conf = float(1 - sd[0] / sd[1]) if len(sd) > 1 and sd[1] > 0 else 1.0
    return {"cluster_id": c, "confianza_relativa": round(conf, 3), "distancias": [round(float(d), 3) for d in dist],
            "z": ((x[0] - STATE["pipeline"].named_steps["scaler"].mean_) / STATE["pipeline"].named_steps["scaler"].scale_)}


def _pct(v: float) -> str:
    return f"{v:.1f}".replace(".", ",")


def _grupo_resumen(c: int) -> dict:
    return {k: STATE["profiles"][c][k] for k in ("cluster_id", "n_municipios", "ipm_medio_interpretativo", "rasgos_distintivos")}


@app.post("/api/v1/predictions", tags=["Segmentación"])
def predict(body: PrivacionesIn, request: Request):
    """Asigna un perfil de privaciones (p. ej. un escenario exploratorio) al grupo más cercano."""
    if (e := _require_model(request)): return e
    feats = STATE["meta"]["features"]
    vals = [getattr(body, f) for f in feats]
    a = _assign(vals)
    means = STATE["pipeline"].named_steps["scaler"].mean_
    zz = a["z"]
    top = np.argsort(-np.abs(zz))[:4]
    return {
        "cluster_id": a["cluster_id"],
        "confianza_relativa": a["confianza_relativa"],
        "distancias": a["distancias"],
        "factores": [{"variable": feats[i], "valor": float(vals[i]), "media_santander": round(float(means[i]), 2),
                      "z": round(float(zz[i]), 2)} for i in top],
        "perfil_grupo": _grupo_resumen(a["cluster_id"]),
        "model_version_id": STATE["meta"]["model_version_id"],
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "correlationId": _cid(request),
        "advertencia": ADVERTENCIA,
    }


# --------------------------------------------------------------------------- Sprint 5
# Catálogo de variables, escenarios exploratorios (RF-13) y priorización orientativa (RF-12).

DIMENSION = {
    "analfabetismo": "Educación", "bajo_logro_educativo": "Educación",
    "barreras_primera_infancia": "Niñez y juventud", "inasistencia_escolar": "Niñez y juventud",
    "rezago_escolar": "Niñez y juventud", "trabajo_infantil": "Niñez y juventud",
    "barreras_acceso_salud": "Salud", "sin_aseguramiento_salud": "Salud",
    "tasa_dependencia": "Trabajo", "trabajo_informal": "Trabajo",
    "hacinamiento_critico": "Vivienda y servicios públicos", "eliminacion_excretas": "Vivienda y servicios públicos",
    "paredes_inadecuadas": "Vivienda y servicios públicos", "pisos_inadecuados": "Vivienda y servicios públicos",
    "sin_fuente_agua_mejorada": "Vivienda y servicios públicos",
}
ETIQUETA = {
    "analfabetismo": "Analfabetismo", "bajo_logro_educativo": "Bajo logro educativo",
    "barreras_primera_infancia": "Barreras a servicios de primera infancia", "inasistencia_escolar": "Inasistencia escolar",
    "rezago_escolar": "Rezago escolar", "trabajo_infantil": "Trabajo infantil",
    "barreras_acceso_salud": "Barreras de acceso a salud", "sin_aseguramiento_salud": "Sin aseguramiento en salud",
    "tasa_dependencia": "Alta tasa de dependencia", "trabajo_informal": "Trabajo informal",
    "hacinamiento_critico": "Hacinamiento crítico", "eliminacion_excretas": "Inadecuada eliminación de excretas",
    "paredes_inadecuadas": "Material inadecuado de paredes", "pisos_inadecuados": "Material inadecuado de pisos",
    "sin_fuente_agua_mejorada": "Sin acceso a fuente de agua mejorada",
}
# Líneas de revisión orientativas por dimensión: señalan qué oferta revisar con la entidad competente;
# no son prescripciones ni sustituyen el análisis sectorial (RF-12, RNF-21).
LINEA_REVISION = {
    "Educación": "Revisar con la secretaría de educación la oferta de alfabetización y de educación para adultos.",
    "Niñez y juventud": "Revisar con educación y bienestar familiar la cobertura de primera infancia, permanencia escolar y prevención del trabajo infantil.",
    "Salud": "Revisar con la secretaría de salud la afiliación al sistema y las barreras de acceso a los servicios.",
    "Trabajo": "Revisar con desarrollo económico la oferta de formalización laboral y generación de ingresos.",
    "Vivienda y servicios públicos": "Revisar con vivienda y con los prestadores de servicios públicos la cobertura de acueducto, saneamiento y mejoramiento de vivienda.",
}
ADVERTENCIA_ESCENARIO = ("Escenario exploratorio y no causal. Compara a qué grupo de municipios se parece un perfil de "
                         "privaciones antes y después de los cambios; no estima el IPM resultante, no predice el efecto de "
                         "una política y no es una recomendación de inversión.")


def _stats() -> dict:
    if "stats" not in STATE:
        feats = STATE["meta"]["features"]
        X = np.array([[float(r[f]) for f in feats] for r in STATE["segments"]])
        STATE["stats"] = {f: {"media": round(float(X[:, i].mean()), 2), "mediana": round(float(np.median(X[:, i])), 2),
                              "p25": round(float(np.percentile(X[:, i], 25)), 2), "p75": round(float(np.percentile(X[:, i], 75)), 2),
                              "min": round(float(X[:, i].min()), 2), "max": round(float(X[:, i].max()), 2)}
                          for i, f in enumerate(feats)}
    return STATE["stats"]


@app.get("/api/v1/variables", tags=["Catálogo"])
def variables(request: Request):
    """Catálogo de las 15 privaciones: etiqueta, dimensión DANE y estadísticas en los 87 municipios."""
    if (e := _require_model(request)): return e
    st = _stats()
    return {"universo": "87 municipios de Santander · CNPV 2018", "unidad": "% de hogares (0-100)",
            "items": [{"variable": f, "etiqueta": ETIQUETA[f], "dimension": DIMENSION[f], "estadisticas": st[f]}
                      for f in STATE["meta"]["features"]]}


class EscenarioIn(BaseModel):
    nombre: str | None = Field(default=None, max_length=120)
    codigo_base: str | None = Field(default=None, description="Código DANE del municipio que sirve de escenario base")
    base: PrivacionesIn | None = Field(default=None, description="Perfil base explícito, si no se usa un municipio")
    cambios: dict[str, float] = Field(description="Variables simulables y su nuevo valor (% de hogares, 0-100)")


@app.post("/api/v1/scenarios", tags=["Escenarios"])
def scenarios(body: EscenarioIn, request: Request):
    """Compara el escenario base contra el escenario alternativo y devuelve diferencias con advertencia no causal."""
    if (e := _require_model(request)): return e
    feats = STATE["meta"]["features"]
    row = None
    if body.codigo_base:
        row = next((r for r in STATE["segments"] if r["codigo"] == body.codigo_base.zfill(5)), None)
        if row is None:
            return error(request, 404, "TERRITORY_NOT_FOUND", "El código no corresponde a un municipio de Santander.",
                         [{"field": "codigo_base", "issue": body.codigo_base}])
        base_vals = [float(row[f]) for f in feats]
    elif body.base is not None:
        base_vals = [float(getattr(body.base, f)) for f in feats]
    else:
        return error(request, 422, "SCENARIO_BASE_REQUIRED", "Indique un municipio base (codigo_base) o un perfil base.",
                     [{"field": "codigo_base", "issue": "requerido si no se envía base"}])
    issues = []
    for k, v in body.cambios.items():
        if k not in feats:
            issues.append({"field": f"cambios.{k}", "issue": "variable no simulable"})
        elif not (0 <= v <= 100):
            issues.append({"field": f"cambios.{k}", "issue": "debe estar entre 0 y 100"})
    if issues:
        return error(request, 422, "INPUT_VALIDATION_ERROR", "El escenario contiene variables o valores inválidos.", issues)
    esc_vals = [float(body.cambios.get(f, base_vals[i])) for i, f in enumerate(feats)]
    changed = [i for i in range(len(feats)) if abs(esc_vals[i] - base_vals[i]) > 1e-9]
    if not changed:
        return error(request, 422, "SCENARIO_NO_CHANGES", "El escenario no modifica ninguna variable respecto a la base.",
                     [{"field": "cambios", "issue": "sin cambios"}])
    a, b = _assign(base_vals), _assign(esc_vals)
    scale = STATE["pipeline"].named_steps["scaler"].scale_
    diffs = [{"variable": feats[i], "etiqueta": ETIQUETA[feats[i]], "dimension": DIMENSION[feats[i]],
              "base": round(base_vals[i], 2), "escenario": round(esc_vals[i], 2),
              "delta": round(esc_vals[i] - base_vals[i], 2), "delta_z": round((esc_vals[i] - base_vals[i]) / scale[i], 2)}
             for i in changed]
    diffs.sort(key=lambda d: -abs(d["delta_z"]))
    acerc = [{"cluster_id": c, "ipm_medio_interpretativo": STATE["profiles"][c]["ipm_medio_interpretativo"],
              "distancia_base": a["distancias"][c], "distancia_escenario": b["distancias"][c],
              "variacion": round(b["distancias"][c] - a["distancias"][c], 3)} for c in range(len(a["distancias"]))]
    if a["cluster_id"] == b["cluster_id"]:
        lectura = (f"El escenario permanece en el grupo {b['cluster_id']}. "
                   f"Su distancia al centro del grupo cambia de {_pct(a['distancias'][a['cluster_id']])} a {_pct(b['distancias'][b['cluster_id']])}.")
    else:
        lectura = (f"Con los cambios, el perfil pasa del grupo {a['cluster_id']} "
                   f"(IPM medio de referencia {_pct(STATE['profiles'][a['cluster_id']]['ipm_medio_interpretativo'])} %) al grupo "
                   f"{b['cluster_id']} (IPM medio de referencia {_pct(STATE['profiles'][b['cluster_id']]['ipm_medio_interpretativo'])} %). "
                   "Indica similitud con otro grupo de municipios, no un cambio estimado del IPM.")
    return {
        "escenario_id": f"esc-{uuid.uuid4().hex[:10]}",
        "nombre": body.nombre,
        "codigo_base": row["codigo"] if row else None,
        "municipio_base": row.get("municipio") if row else None,
        "ipm_base_observado": float(row["ipm"]) if row else None,
        "base": {"cluster_id": a["cluster_id"], "confianza_relativa": a["confianza_relativa"], "distancias": a["distancias"],
                 "perfil_grupo": _grupo_resumen(a["cluster_id"])},
        "escenario": {"cluster_id": b["cluster_id"], "confianza_relativa": b["confianza_relativa"], "distancias": b["distancias"],
                      "perfil_grupo": _grupo_resumen(b["cluster_id"])},
        "cambio_de_grupo": a["cluster_id"] != b["cluster_id"],
        "variables_modificadas": len(changed),
        "diferencias": diffs,
        "distancias_por_grupo": acerc,
        "lectura": lectura,
        "model_version_id": STATE["meta"]["model_version_id"],
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "correlationId": _cid(request),
        "advertencia": ADVERTENCIA_ESCENARIO,
    }


@app.get("/api/v1/priorities", tags=["Priorización"])
def priorities(request: Request, cluster_id: int | None = None, limit: int = 87):
    """Priorización orientativa: ordena municipios por número de privaciones críticas (sobre el percentil 75
    de Santander) y, a igualdad, por IPM observado. Regla transparente y no vinculante."""
    if (e := _require_model(request)): return e
    feats = STATE["meta"]["features"]
    st = _stats()
    items = []
    for r in STATE["segments"]:
        c = int(r["cluster_id"])
        if cluster_id is not None and c != cluster_id:
            continue
        focos = [{"variable": f, "etiqueta": ETIQUETA[f], "dimension": DIMENSION[f], "valor": float(r[f]),
                  "p75_santander": st[f]["p75"], "mediana_santander": st[f]["mediana"]}
                 for f in feats if float(r[f]) > st[f]["p75"]]
        dims: dict[str, int] = {}
        for fo in focos:
            dims[fo["dimension"]] = dims.get(fo["dimension"], 0) + 1
        tam = {d: sum(1 for f in feats if DIMENSION[f] == d) for d in dims}
        exc = {d: sum(fo["valor"] - fo["p75_santander"] for fo in focos if fo["dimension"] == d) for d in dims}
        # Dimensión más comprometida: mayor proporción de sus privaciones en nivel crítico; a igualdad, mayor exceso sobre el p75.
        dims_orden = sorted(dims.items(), key=lambda kv: (-kv[1] / tam[kv[0]], -exc[kv[0]]))
        items.append({"codigo": r["codigo"], "municipio": r.get("municipio"), "ipm": float(r["ipm"]), "cluster_id": c,
                      "privaciones_criticas": len(focos), "dimensiones": [{"dimension": d, "privaciones": n, "de": tam[d]} for d, n in dims_orden],
                      "focos": sorted(focos, key=lambda fo: -(fo["valor"] - fo["p75_santander"])),
                      "lineas_revision": [{"dimension": d, "texto": LINEA_REVISION[d]} for d, _ in dims_orden[:2]]})
    items.sort(key=lambda it: (-it["privaciones_criticas"], -it["ipm"]))
    for i, it in enumerate(items, 1):
        it["posicion"] = i
    return {"model_version_id": STATE["meta"]["model_version_id"], "total": len(items), "items": items[:max(1, limit)],
            "criterio": ("Privación crítica: valor del municipio por encima del percentil 75 de los 87 municipios de Santander. "
                         "Orden: más privaciones críticas primero; a igualdad, mayor IPM observado (CNPV 2018). "
                         "Línea de revisión: dimensión con mayor proporción de privaciones críticas."),
            "advertencia": ("Priorización orientativa y no vinculante (RF-12). Las líneas de revisión señalan qué oferta "
                            "revisar con la entidad competente; no son prescripciones ni sustituyen el análisis sectorial.")}
