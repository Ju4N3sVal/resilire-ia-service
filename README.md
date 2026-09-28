# RESILIRE · Microservicio de IA (Sprint 3)

Segmentación territorial de vulnerabilidad multidimensional de los 87 municipios de Santander (DANE, CNPV 2018), según la configuración de referencia del documento PI2-RES-S2-04. La configuración es K-means con k de 2 a 5, semilla 42 y n_init de 20. Los insumos son las 15 privaciones estandarizadas. El IPM y el código municipal quedan excluidos como predictores.

## 1. Entrenar (reproducible)

```bash
pip install -r requirements.txt
# Descargar el anexo oficial a data/raw/:
# https://www.dane.gov.co/files/investigaciones/condiciones_vida/pobreza/2018/informacion-censal/anexo-censal-pobreza-municipal-2018.xlsx
python -m training.train --input data/raw/anexo-censal-pobreza-municipal-2018.xlsx --out artifacts
```

El entrenamiento genera los artefactos de la sección 8 del documento del Sprint 2: `data_processed_v1.csv`, `data_dictionary_v1.csv`, `quality_report_v1.csv`, `baseline_model_v1.joblib`, `input_schema_v1.json` y `metrics_baseline_v1.json`. También produce `segments_v1.csv`, `profiles_v1.json`, `importance_v1.json` y `correlations_v1.json`.

## 2. Ejecutar la API

```bash
uvicorn app.main:app --reload     # Swagger en http://127.0.0.1:8000/docs
```

| Método | Ruta | Propósito |
|---|---|---|
| GET | /health | Disponibilidad del servicio y versión del modelo |
| GET | /api/v1/model | Ficha del modelo y métricas |
| GET | /api/v1/segments | Municipios con su grupo (`?cluster_id=`) |
| GET | /api/v1/segments/{codigo} | Grupo y perfil de un municipio (ej. 68001) |
| GET | /api/v1/profiles | Perfil promedio de cada grupo |
| GET | /api/v1/importance | Impacto de variables (ANOVA F, permutación) |
| GET | /api/v1/correlations | Correlaciones descriptivas con el IPM |
| POST | /api/v1/predictions | Asigna un perfil de 15 privaciones a un grupo |

Cada respuesta lleva la cabecera `X-Correlation-ID`. Los errores usan el formato uniforme `{code, message, correlationId, details}`.

## 3. Desplegar

El servicio se despliega en Render con `render.yaml` (Docker, plan gratuito) o en cualquier servicio que ejecute contenedores. CORS permite `resilire-web.web.app` por defecto; se cambia con `RESILIRE_CORS_ORIGINS`.

Toda salida es orientativa y no vinculante. No predice el IPM, no es causal y no aplica a hogares o personas.
