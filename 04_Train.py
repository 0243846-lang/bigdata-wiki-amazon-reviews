# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  04_Train  —  Pipeline ML: TF-IDF + Regresión Logística                    ║
# ║  Catalog : big_datawiki  |  Schema : inversion_schema                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# PRERREQUISITO (ejecutar en celda SQL antes de correr este notebook):
#   CREATE VOLUME IF NOT EXISTS big_datawiki.inversion_schema.models;

from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
import os

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG          = "big_datawiki"
SCHEMA           = "inversion_schema"
VOLUME_PATH      = f"/Volumes/{CATALOG}/{SCHEMA}/models"
REGISTERED_MODEL = f"{CATALOG}.{SCHEMA}.review_classifier"
LABEL_COL        = "label_idx"
FEATURE_COL      = "Review"
NUM_FEATURES     = 262144

os.environ["MLFLOW_DFS_TMP"] = VOLUME_PATH

# ── Cargar datos ──────────────────────────────────────────────────────────────
df_train = spark.table(f"{CATALOG}.{SCHEMA}.gold_train").select(FEATURE_COL, LABEL_COL)
df_test  = spark.table(f"{CATALOG}.{SCHEMA}.gold_test").select(FEATURE_COL, LABEL_COL)

print(f"train: {df_train.count():,} | test: {df_test.count():,}")

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline = Pipeline(stages=[
    Tokenizer(inputCol=FEATURE_COL, outputCol="tokens"),
    StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens"),
    HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=NUM_FEATURES),
    IDF(inputCol="raw_features", outputCol="features", minDocFreq=5),
    LogisticRegression(
        featuresCol="features",
        labelCol=LABEL_COL,
        maxIter=20,
        regParam=0.01,
        elasticNetParam=0.0,
    ),
])

# ── Entrenamiento con MLflow tracking ─────────────────────────────────────────
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/bigdata_wiki/review_classifier")

with mlflow.start_run(run_name="tfidf_logreg_v1"):

    mlflow.log_params({
        "num_features": NUM_FEATURES,
        "max_iter":     20,
        "reg_param":    0.01,
        "elastic_net":  0.0,
        "min_doc_freq": 5,
    })

    print("Entrenando pipeline...")
    model = pipeline.fit(df_train)

    print("Evaluando en test...")
    preds = model.transform(df_test)

    accuracy = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="f1"
    ).evaluate(preds)
    auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(preds)

    mlflow.log_metrics({"test_accuracy": accuracy, "test_f1": f1, "test_auc_roc": auc})

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    # Registrar modelo en Unity Catalog
    input_example = df_train.limit(5).toPandas()
    mlflow.spark.log_model(
        model,
        artifact_path="spark_model",
        dfs_tmpdir=VOLUME_PATH,
        registered_model_name=REGISTERED_MODEL,
        input_example=input_example,
    )
    print(f"Modelo registrado como: {REGISTERED_MODEL}")

# ── Análisis de predicciones ──────────────────────────────────────────────────
print("Matriz de confusión (test):")
display(
    preds.groupBy(LABEL_COL, "prediction")
         .count()
         .orderBy(LABEL_COL, "prediction")
)

print("Muestra de predicciones:")
display(
    preds.select(FEATURE_COL, LABEL_COL, "prediction", "probability")
         .limit(10)
)

print("Errores de clasificación (muestra):")
display(
    preds.filter(col(LABEL_COL) != col("prediction"))
         .select(FEATURE_COL, LABEL_COL, "prediction")
         .limit(10)
)
