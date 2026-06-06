# 🛒 Amazon Reviews — Clasificación con Pipeline Medallion en Databricks

Pipeline completo de datos (Bronze → Silver → Gold) + modelo de Machine Learning con Spark MLlib para clasificar reseñas de Amazon como **positivas (1)** o **negativas (0)**.

---

## ⚙️ Configuración Inicial (Desde Cero)

### 1. Crear el Catalog y Schema en Unity Catalog

1. En Databricks, ve al menú izquierdo y haz clic en **Catalog** (ícono de base de datos).
2. Haz clic en **+ Add** → **Add a catalog**.
3. Ponle de nombre: `big_datawiki`.
4. Dentro del catalog, haz clic en **+ Add** → **Add a schema**.
5. Ponle de nombre: `inversion_schema`.

### 2. Subir los archivos del Dataset (train y test)

1. Ve a **Catalog** → `big_datawiki` → `inversion_schema`.
2. Haz clic en **Create table** (botón azul arriba a la derecha).
3. Selecciona **Upload files**.
4. Sube `train.csv` → tabla `train`.
5. Repite para `test.csv` → tabla `test`.
6. Verifica que ambas aparezcan con columnas `Review` (string) y `Label` (bigint).

### 3. Crear el Volume para MLflow

Ejecuta en una celda SQL antes del notebook de entrenamiento:

```sql
CREATE VOLUME IF NOT EXISTS big_datawiki.inversion_schema.models;
```

---

## 📌 Descripción del Proyecto

Arquitectura **Medallion** sobre Databricks con Unity Catalog. Se procesan reseñas de Amazon a través de Bronze, Silver y Gold, y se entrena un modelo TF-IDF + Regresión Logística desplegado como endpoint REST.

---

## 🗂️ Estructura del Proyecto

```
bigdata_wiki/
├── 01_Bronce.py  — Ingesta y metadatos
├── 02_Silver.py  — Limpieza y feature engineering
├── 03_Gold.py    — Enriquecimiento y análisis exploratorio
└── 04_Train.py   — Entrenamiento, evaluación y registro del modelo
```

### Tablas en Unity Catalog

| Tabla | Capa | Descripción |
|---|---|---|
| train | Raw | Dataset original subido vía UI |
| test | Raw | Dataset de test original |
| bronze_train | Bronze | Train + metadata de ingesta |
| bronze_test | Bronze | Test + metadata de ingesta |
| silver_train | Silver | Limpio + features de texto |
| silver_test | Silver | Limpio + features de texto |
| gold_train | Gold | ML-ready con buckets de longitud |
| gold_test | Gold | ML-ready con buckets de longitud |

---

## 🔄 Pipeline Paso a Paso

### 🥉 Bronze — 01_Bronce.py
Lee tablas raw y agrega `_split` y `_ingestion_ts`. Resultado: 3.6M filas train, 400K test.

### 🥈 Silver — 02_Silver.py
Elimina nulos, duplicados y HTML. Agrega `review_length`, `word_count`, `label_idx` (0-indexado).

### 🥇 Gold — 03_Gold.py
Agrega `length_bucket` (short/medium/long/very_long) e `is_long_review`. Genera EDA.

### 🤖 Train — 04_Train.py
Pipeline: Tokenizer → StopWordsRemover → HashingTF(262,144) → IDF → LogisticRegression.
Métricas en MLflow: `test_accuracy`, `test_f1`, `test_auc_roc`.
Modelo registrado como `big_datawiki.inversion_schema.review_classifier`.

---

## 🚀 Endpoint (Postman)

| Campo | Valor |
|---|---|
| Método | POST |
| URL | https://dbc-0b0dcebc-1ae3.cloud.databricks.com/serving-endpoints/amazon_final/invocations |
| Authorization | Bearer Token |
| Content-Type | application/json |

**Body:**
```json
{
  "dataframe_records": [
    {"Review": "This product is absolutely amazing!", "label_idx": 0},
    {"Review": "Terrible product, completely broken.", "label_idx": 0}
  ]
}
```
**Respuesta:** `{"predictions": [1.0, 0.0]}` — 1.0 = positiva · 0.0 = negativa

---

## 🛠️ Tecnologías
Databricks · PySpark · Spark MLlib · MLflow · Unity Catalog · Delta Lake · Postman

## 📊 Dataset
Reseñas de Amazon — 3.6M train / 400K test — clasificación binaria de sentimiento.

## 👤 Autor
David — Universidad Panamericana · Big Data
