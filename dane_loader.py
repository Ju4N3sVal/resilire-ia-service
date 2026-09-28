"""
Lector del anexo censal DANE — Medida de pobreza multidimensional municipal
de fuente censal (CNPV 2018).

Archivo oficial:
https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/2018/informacion-censal/anexo-censal-pobreza-municipal-2018.xlsx

El anexo publica varias hojas con encabezados desplazados. Este lector
busca en cada hoja la fila de encabezado, identifica las columnas por
palabras clave y une las hojas por el código territorial DANE
(regla de limpieza 1: el código se conserva como texto).
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# 15 privaciones del IPM (5 dimensiones) + nombre canónico
PRIVACIONES = {
    "analfabetismo": ("Educación", ["analfabetismo"]),
    "bajo_logro_educativo": ("Educación", ["bajo logro"]),
    "barreras_primera_infancia": ("Niñez y juventud", ["primera infancia"]),
    "inasistencia_escolar": ("Niñez y juventud", ["inasistencia"]),
    "rezago_escolar": ("Niñez y juventud", ["rezago"]),
    "trabajo_infantil": ("Niñez y juventud", ["trabajo infantil"]),
    "barreras_acceso_salud": ("Salud", ["barreras de acceso a servicios de salud", "barreras de acceso a salud", "acceso a servicios de salud", "barreras de acceso"]),
    "sin_aseguramiento_salud": ("Salud", ["aseguramiento"]),
    "tasa_dependencia": ("Trabajo", ["dependencia"]),
    "trabajo_informal": ("Trabajo", ["trabajo informal", "informal"]),
    "hacinamiento_critico": ("Vivienda y servicios", ["hacinamiento"]),
    "eliminacion_excretas": ("Vivienda y servicios", ["excretas"]),
    "paredes_inadecuadas": ("Vivienda y servicios", ["paredes"]),
    "pisos_inadecuados": ("Vivienda y servicios", ["pisos"]),
    "sin_fuente_agua_mejorada": ("Vivienda y servicios", ["fuente de agua", "agua mejorada"]),
}

FEATURES = list(PRIVACIONES.keys())


def _norm(text) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_header_row(raw: pd.DataFrame, max_rows: int = 30) -> int | None:
    for i in range(min(max_rows, len(raw))):
        row = [_norm(v) for v in raw.iloc[i].tolist()]
        joined = " | ".join(row)
        if ("codigo" in joined or "cod" in joined) and "municipio" in joined:
            return i
    return None


def _match_column(columns: list[str], keywords: list[str]) -> str | None:
    for kw in keywords:
        for col in columns:
            if kw in _norm(col):
                return col
    return None


def _table(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    df = df[df["ID"].notna()]
    df["ID"] = df["ID"].astype(str).str.strip().str.zfill(5)
    return df[df["ID"].str.fullmatch(r"\d{5}")]


def _load_official(sheets: dict) -> pd.DataFrame:
    ipm = _table(sheets["2_IPM Mpio y Dpt"], 0)[["ID", "Municipio", "IPM Municipal"]]
    ipm.columns = ["codigo", "municipio", "ipm"]

    dom = _table(sheets["4_IPM Mpio dominios"], 0)
    dom = dom.iloc[:, [0, 3, 4]]
    dom.columns = ["codigo", "ipm_cabecera", "ipm_rural"]

    raw6 = sheets["6_Privaciones IPM Dpt-Mpio"]
    h = _find_header_row(raw6.rename(columns=str).astype(object).fillna(""), 5)
    h = 1 if h is None else h
    priv = _table(raw6, h)
    cols = list(priv.columns)
    out = pd.DataFrame({"codigo": priv["ID"]})
    used = set()
    for feat, (_, kws) in PRIVACIONES.items():
        col = _match_column([c for c in cols if c not in used and c not in ("ID", "Municipio")], kws)
        if col is None:
            raise ValueError(f"No se encontró la columna para {feat}")
        used.add(col)
        out[feat] = priv[col].values
        out.attrs.setdefault("mapeo", {})[feat] = col

    df = ipm.merge(dom, on="codigo", how="left").merge(out, on="codigo", how="left")
    df.attrs["mapeo"] = out.attrs.get("mapeo", {})
    return df


def load_anexo(path: str) -> pd.DataFrame:
    """Devuelve un DataFrame con columnas: codigo, departamento, municipio,
    ipm y las 15 privaciones (porcentaje de hogares, 0-100), a nivel total
    municipal. Si una hoja distingue cabecera/rural se toma el total."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype={"codigo": str})
        missing = [c for c in ["codigo", "municipio", "ipm", *FEATURES] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV sin columnas requeridas: {missing}")
        return df

    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)

    # Formato oficial del anexo DANE 2018 (hojas 2, 4 y 6)
    if {"2_IPM Mpio y Dpt", "6_Privaciones IPM Dpt-Mpio"} <= set(sheets):
        return _load_official(sheets)

    frames = []
    for name, raw in sheets.items():
        h = _find_header_row(raw)
        if h is None:
            continue
        df = raw.iloc[h + 1:].copy()
        df.columns = [str(c) for c in raw.iloc[h].tolist()]
        df = df.loc[:, [c for c in df.columns if c and c != "nan"]]
        cols = list(df.columns)
        code_col = next((c for c in cols if _norm(c).startswith("cod") and "mun" in _norm(c)), None) \
            or next((c for c in cols if _norm(c).startswith("cod")), None)
        mun_col = next((c for c in cols if _norm(c) in ("municipio", "nombre municipio")), None) \
            or _match_column(cols, ["municipio"])
        if code_col is None:
            continue
        out = pd.DataFrame({"codigo": df[code_col]})
        if mun_col:
            out["municipio"] = df[mun_col]
        dep_col = _match_column(cols, ["departamento"])
        if dep_col and dep_col != mun_col:
            out["departamento"] = df[dep_col]
        # IPM total
        ipm_col = next((c for c in cols if _norm(c) in ("total", "ipm", "ipm total", "incidencia total")), None)
        if ipm_col is None:
            ipm_col = _match_column(cols, ["ipm"])
        if ipm_col is not None and not any(k in _norm(ipm_col) for k in ("cabecera", "rural", "centros")):
            out["ipm"] = df[ipm_col]
        for feat, (_, kws) in PRIVACIONES.items():
            col = _match_column(cols, kws)
            if col is not None:
                out[feat] = df[col]
        out["_sheet"] = name
        frames.append(out)

    if not frames:
        raise ValueError("No se identificaron hojas con código y municipio en el anexo.")

    merged = None
    for f in frames:
        f = f.dropna(subset=["codigo"])
        f["codigo"] = f["codigo"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(5)
        f = f[f["codigo"].str.fullmatch(r"\d{5}")]
        f = f.drop(columns=["_sheet"])
        if merged is None:
            merged = f
        else:
            new_cols = [c for c in f.columns if c not in merged.columns or c == "codigo"]
            merged = merged.merge(f[new_cols], on="codigo", how="outer")
    return merged
