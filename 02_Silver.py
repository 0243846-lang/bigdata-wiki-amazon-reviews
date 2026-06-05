from pyspark.sql.functions import (
    col, trim, regexp_replace, length, size, split,
    when, current_timestamp, lit
)

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG = "big_datawiki"
SCHEMA  = "inversion_schema"

SOURCES = {
    "bronze_train": "silver_train",
    "bronze_test":  "silver_test",
}

# ── Cleaning function ─────────────────────────────────────────────────────────
def clean_reviews(df):
    return (
        df
        .filter(col("Review").isNotNull() & (trim(col("Review")) != ""))
        .filter(col("Label").isNotNull())
        .dropDuplicates(["Review", "Label"])
        .withColumn(
            "Review",
            trim(regexp_replace(
                regexp_replace(col("Review"), r"<[^>]+>", " "),
                r"\s+", " "
            ))
        )
        .withColumn("review_length", length(col("Review")))
        .withColumn("word_count",    size(split(col("Review"), r"\s+")))
        .withColumn("label_idx",     (col("Label") - lit(1)).cast("int"))
        .withColumn("_silver_ts",    current_timestamp())
    )

# ── Process and save ──────────────────────────────────────────────────────────
for source, target in SOURCES.items():
    df_silver = clean_reviews(spark.table(f"{CATALOG}.{SCHEMA}.{source}"))
    df_silver.write \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{target}")
    print(f"Saved → {CATALOG}.{SCHEMA}.{target}")

# ── Validation ────────────────────────────────────────────────────────────────
print("\n── Silver counts ──")
for target in SOURCES.values():
    df    = spark.table(f"{CATALOG}.{SCHEMA}.{target}")
    total = df.count()
    print(f"\n {target}: {total:,} rows")
    for row in df.groupBy("label_idx").count().orderBy("label_idx").collect():
        print(f"   label_idx={row['label_idx']} → {row['count']:,} ({row['count']/total*100:.1f}%)")

print("\n── Length stats ──")
display(
    spark.table(f"{CATALOG}.{SCHEMA}.silver_train")
         .select("review_length", "word_count")
         .summary("min", "25%", "50%", "75%", "max", "mean")
)
display(spark.table(f"{CATALOG}.{SCHEMA}.silver_train").limit(5))
