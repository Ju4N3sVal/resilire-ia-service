"""
RESILIRE — Microservicio de IA (Sprint 3)

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

app = FastAPI(title="RESILIRE · Microservicio de IA", version="1.0.0",
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
            "cluster_id": c, "perfil": STATE["profiles"][c], "model_version_id": STATE["meta"]["model_version_id"],
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


@app.post("/api/v1/predictions", tags=["Segmentación"])
def predict(body: PrivacionesIn, request: Request):
    """Asigna un perfil de privaciones (p. ej. un escenario exploratorio) al grupo más cercano."""
    if (e := _require_model(request)): return e
    feats = STATE["meta"]["features"]
    x = np.array([[getattr(body, f) for f in feats]], dtype=float)
    pipe = STATE["pipeline"]
    z = pipe.named_steps["scaler"].transform(x)
    dist = pipe.named_steps["kmeans"].transform(z)[0]
    c = int(np.argmin(dist))
    sorted_d = np.sort(dist)
    confianza = float(1 - sorted_d[0] / sorted_d[1]) if len(sorted_d) > 1 and sorted_d[1] > 0 else 1.0
    means = np.array(pipe.named_steps["scaler"].mean_)
    scale = np.array(pipe.named_steps["scaler"].scale_)
    zz = (x[0] - means) / scale
    top = np.argsort(-np.abs(zz))[:4]
    return {
        "cluster_id": c,
        "confianza_relativa": round(confianza, 3),
        "distancias": [round(float(d), 3) for d in dist],
        "factores": [{"variable": feats[i], "valor": float(x[0][i]), "media_santander": round(float(means[i]), 2),
                      "z": round(float(zz[i]), 2)} for i in top],
        "perfil_grupo": {k: STATE["profiles"][c][k] for k in ("cluster_id", "n_municipios", "ipm_medio_interpretativo", "rasgos_distintivos")},
        "model_version_id": STATE["meta"]["model_version_id"],
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "correlationId": _cid(request),
        "advertencia": ADVERTENCIA,
    }
