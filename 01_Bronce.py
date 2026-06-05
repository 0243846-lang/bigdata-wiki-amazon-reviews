from pyspark.sql.functions import col, current_timestamp, lit, length

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG = "big_datawiki"
SCHEMA  = "inversion_schema"

# UI-uploaded tables → bronze destination
SPLITS = {
    "train": "bronze_train",
    "test":  "bronze_test",
}

# ── Add metadata + save as bronze ─────────────────────────────────────────────
for source, target in SPLITS.items():
    df = spark.table(f"{CATALOG}.{SCHEMA}.{source}") \
              .withColumn("_split",        lit(source)) \
              .withColumn("_ingestion_ts", current_timestamp())

    df.write \
      .mode("overwrite") \
      .option("overwriteSchema", "true") \
      .saveAsTable(f"{CATALOG}.{SCHEMA}.{target}")

    print(f"Saved → {CATALOG}.{SCHEMA}.{target}")

# ── Validation ────────────────────────────────────────────────────────────────
print("\n── Row counts ──")
for target in SPLITS.values():
    df    = spark.table(f"{CATALOG}.{SCHEMA}.{target}")
    total = df.count()
    nulls = df.filter(col("Label").isNull()).count()
    empty = df.filter(col("Review").isNull() | (length(col("Review")) == 0)).count()
    print(f"  {target:<15} | rows: {total:>12,} | null labels: {nulls:>6,} | empty reviews: {empty:>6,}")

print("\n── Sample rows ──")
for target in SPLITS.values():
    print(f"\n  {target}:")
    display(spark.table(f"{CATALOG}.{SCHEMA}.{target}").limit(5))
