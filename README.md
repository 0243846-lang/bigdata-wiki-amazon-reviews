# 🛒 Amazon Reviews — Clasificación con Pipeline Medallion en Databricks

Pipeline completo de datos (Bronze → Silver → Gold) + modelo de Machine Learning con Spark MLlib para clasificar reseñas de Amazon como **positivas (1)** o **negativas (0)**.

---

## 📌 Descripción del Proyecto

Este proyecto implementa una arquitectura **Medallion** sobre Databricks usando el catálogo Unity Catalog (`big_datawiki.inversion_schema`). Se parte de reseñas de Amazon crudas, se limpian y enriquecen a través de las capas Bronze, Silver y Gold, y finalmente se entrena un modelo de clasificación de texto con TF-IDF + Regresión Logística.

El modelo queda desplegado como **endpoint REST** en Databricks Model Serving y puede ser consumido desde Postman u otras herramientas HTTP.

---

## 🗂️ Estructura del Proyecto

```
bigdata_wiki/
├── 01_Bronce    — Ingesta y metadatos
├── 02_Silver    — Limpieza y feature engineering
├── 03_Gold      — Enriquecimiento y análisis exploratorio
└── 04_train     — Entrenamiento, evaluación y registro del modelo
```

### Tablas en Unity Catalog (`big_datawiki.inversion_schema`)

| Tabla | Capa | Descripción |
|---|---|---|
| `train` | Raw | Dataset original subido vía UI |
| `test` | Raw | Dataset de test original |
| `bronze_train` | Bronze | Train + metadata de ingesta |
| `bronze_test` | Bronze | Test + metadata de ingesta |
| `silver_train` | Silver | Limpio + features de texto |
| `silver_test` | Silver | Limpio + features de texto |
| `gold_train` | Gold | ML-ready con buckets de longitud |
| `gold_test` | Gold | ML-ready con buckets de longitud |

---

## 🔄 Pipeline Paso a Paso

### 🥉 Capa Bronze — `01_Bronce`

Lee las tablas raw (`train` y `test`) y les agrega metadatos de ingesta.

**Columnas añadidas:**
- `_split` — indica si el registro es `"train"` o `"test"`
- `_ingestion_ts` — timestamp del momento de ingesta

**Esquema resultante:**

| Columna | Tipo | Descripción |
|---|---|---|
| `Review` | string | Texto de la reseña |
| `Label` | bigint | Etiqueta (1=negativo, 2=positivo) |
| `_split` | string | Partición de origen |
| `_ingestion_ts` | timestamp | Momento de ingesta |

**Volumen:** bronze_train: 3,600,000 filas · bronze_test: 400,000 filas

---

### 🥈 Capa Silver — `02_Silver`

Limpia y enriquece los datos de bronze.

**Transformaciones:**
1. Elimina registros nulos o vacíos
2. Deduplica por `(Review, Label)`
3. Elimina tags HTML y colapsa espacios
4. Agrega features de longitud

**Columnas añadidas:**
- `review_length` — número de caracteres
- `word_count` — número de palabras
- `label_idx` — etiqueta 0-indexada para Spark ML (Label - 1)
- `_silver_ts` — timestamp de procesamiento

---

### 🥇 Capa Gold — `03_Gold`

Prepara los datos para ML y genera análisis exploratorio.

**Columnas añadidas:**
- `length_bucket` — categoría: `short` (<100), `medium` (<500), `long` (<1000), `very_long`
- `is_long_review` — flag binario si review_length >= 500
- `_gold_ts` — timestamp de procesamiento

**Análisis generado:** distribución de etiquetas, estadísticas de longitud por etiqueta, distribución de buckets.

---

### 🤖 Entrenamiento — `04_train`

Entrena un pipeline de clasificación de texto y registra el modelo en Unity Catalog.

**Pipeline de Spark ML:**

```
Review → Tokenizer → StopWordsRemover → HashingTF (262,144) → IDF → LogisticRegression → predicción
```

**Prerrequisito — crear el Volume:**
```sql
CREATE VOLUME IF NOT EXISTS big_datawiki.inversion_schema.models;
```

**Métricas en MLflow:** `test_accuracy`, `test_f1`, `test_auc_roc`

El modelo se registra como `big_datawiki.inversion_schema.review_classifier` en **Catalog → Models**.

---

## 🚀 Cómo Usar el Endpoint (Postman)

**Configuración:**

| Campo | Valor |
|---|---|
| Método | POST |
| URL | `https://dbc-0b0dcebc-1ae3.cloud.databricks.com/serving-endpoints/amazon_final/invocations` |
| Authorization | Bearer Token |
| Content-Type | application/json |

**Body (raw JSON):**
```json
{
  "dataframe_records": [
    {"Review": "This product is absolutely amazing, I love it!", "label_idx": 0},
    {"Review": "Terrible product, completely broken on arrival.", "label_idx": 0},
    {"Review": "It works fine, nothing special but does the job.", "label_idx": 0}
  ]
}
```

**Respuesta:**
```json
{
  "predictions": [1.0, 0.0, 1.0]
}
```

> `1.0` = reseña **positiva** | `0.0` = reseña **negativa**

> El campo `label_idx` es requerido por el schema pero no afecta la predicción — usa `0` en todos los registros.

---

## 🛠️ Tecnologías

- **Databricks** (Free Edition) — Serverless Compute
- **Apache Spark / PySpark** — procesamiento distribuido
- **Spark MLlib** — pipeline de ML (TF-IDF + Logistic Regression)
- **MLflow** — tracking y registro de experimentos
- **Unity Catalog** — gestión de tablas, modelos y volumes
- **Delta Lake** — formato de almacenamiento
- **Postman** — pruebas del endpoint REST

---

## 📊 Dataset

- **Fuente:** Reseñas de Amazon (clasificación binaria de sentimiento)
- **Train:** 3,600,000 reseñas
- **Test:** 400,000 reseñas
- **Clases:** 0 = negativa · 1 = positiva

---

## 👤 Autor

**David** — Universidad Panamericana · Big Data
