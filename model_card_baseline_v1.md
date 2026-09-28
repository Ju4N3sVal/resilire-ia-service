# Model card — baseline_kmeans_v1

**Propósito.** Segmentar los 87 municipios de Santander por similitud de su perfil de privaciones (CNPV 2018) para apoyar la exploración territorial. No predice el IPM, no es causal y no aplica a hogares o personas.

**Datos.** DANE · Medida de pobreza multidimensional municipal de fuente censal (CNPV 2018). Archivo `anexo-censal-pobreza-municipal-2018.xlsx` (sha256 a3ac3a60498ad1b1…). 87 municipios; 0 faltantes; 0 códigos duplicados.

**Configuración.** StandardScaler + K-means; k evaluado de 2 a 5; semilla 42; n_init 20; k seleccionado = 5 por máximo silhouette. Insumos: 15 privaciones. Excluidos: código municipal e IPM (matriz de fuga).

**Métricas.**

| k | Silhouette | Inercia | Davies-Bouldin | Calinski-Harabasz | Tamaños |
|---|---|---|---|---|---|
| 2 | 0.1653 | 1055.64 | 1.7996 | 20.08 | [57, 30] |
| 3 | 0.1674 | 914.9 | 1.8692 | 17.91 | [44, 28, 15] |
| 4 | 0.1733 | 816.25 | 1.6335 | 16.57 | [43, 18, 15, 11] |
| 5 | 0.1782 | 720.72 | 1.4779 | 16.62 | [39, 20, 11, 10, 7] |

Estabilidad (bootstrap, ARI): media 0.498 (p5 0.2402 – p95 0.7926).

Modelo sustituto explicativo (árbol, profundidad 3, StratifiedKFold(5), semilla 42): accuracy 0.7013 ± 0.0415, precisión macro 0.6999, recall macro 0.6519, F1 macro 0.6426.

**Limitaciones.** Línea base 2018; unidad municipal (sesgo ecológico si se lee a nivel individual); 87 observaciones; la separación entre grupos es moderada, por lo que la pertenencia de municipios en el borde entre grupos debe leerse con cautela.

**Uso.** Orientativo y no vinculante. La activación del modelo requiere aprobación del Comité IA / Product Owner.
