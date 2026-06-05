# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  03_Gold  —  Feature engineering, balanceo y tablas ML-ready               ║
# ║  Catalog : big_datawiki  |  Schema : inversion_schema                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from pyspark.sql.functions import (
    col, length, size, split, when, current_timestamp,
    count, avg, stddev, min as fmin, max as fmax, percentile_approx
)

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG = "big_datawiki"
SCHEMA  = "inversion_schema"

SOURCES = {
    "silver_train": "gold_train",
    "silver_test":  "gold_test",
}

# ── Función de enriquecimiento Gold ──────────────────────────────────────────
def enrich_gold(df):
    """
    Enriquece el DataFrame silver para dejarlo ML-ready:
    - Añade buckets de longitud de reseña
    - Añade feature binaria 'is_long_review'
    - Selecciona y ordena columnas finales
    - Marca timestamp gold
    """
    return (
        df
        .withColumn(
            "length_bucket",
            when(col("review_length") <  100, "short")
            .when(col("review_length") <  500, "medium")
            .when(col("review_length") < 1000, "long")
            .otherwise("very_long")
        )
        .withColumn("is_long_review",
            (col("review_length") >= 500).cast("int")
        )
        .withColumn("_gold_ts", current_timestamp())
        .select(
            "Review",
            "Label",
            "label_idx",
            "review_length",
            "word_count",
            "length_bucket",
            "is_long_review",
            "_split",
            "_ingestion_ts",
            "_silver_ts",
            "_gold_ts",
        )
    )

# ── Procesar y guardar ────────────────────────────────────────────────────────
for source, target in SOURCES.items():
    df_silver = spark.table(f"{CATALOG}.{SCHEMA}.{source}")
    df_gold   = enrich_gold(df_silver)

    df_gold.write \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{target}")

    print(f"Saved -> {CATALOG}.{SCHEMA}.{target}")

# ── Análisis exploratorio de gold_train ──────────────────────────────────────
df_gt = spark.table(f"{CATALOG}.{SCHEMA}.gold_train")
total = df_gt.count()

print(f"gold_train: {total:,} filas")

display(
    df_gt.groupBy("label_idx", "Label")
         .count()
         .withColumn("pct", (col("count") / total * 100).cast("decimal(5,2)"))
         .orderBy("label_idx")
)

display(
    df_gt.groupBy("label_idx", "length_bucket")
         .count()
         .orderBy("label_idx", "length_bucket")
)

display(
    df_gt.groupBy("label_idx")
         .agg(
             count("*").alias("n"),
             avg("review_length").alias("avg_length"),
             stddev("review_length").alias("std_length"),
             percentile_approx("review_length", 0.5).alias("median_length"),
             avg("word_count").alias("avg_words"),
             percentile_approx("word_count", 0.5).alias("median_words"),
         )
         .orderBy("label_idx")
)

display(df_gt.limit(5))
