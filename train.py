
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml import Pipeline
from pyspark.ml.feature import (Tokenizer, StopWordsRemover, HashingTF, IDF,
                                 StringIndexer, IndexToString)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from functools import reduce

CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else "data/mendeley.csv"
MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "model/genre_model"

spark = SparkSession.builder \
    .appName("MusicGenreClassifier") \
    .master("local[2]") \
    .config("spark.driver.memory", "3g") \
    .config("spark.executor.memory", "3g") \
    .config("spark.driver.maxResultSize", "1g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "8") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(CSV_PATH, header=True, inferSchema=True)

# Normalize column names
df = df.toDF(*[c.lower().strip().replace("/", "_").replace(" ", "_") for c in df.columns])
df = df.withColumn("genre", lower(col("genre")))

# Keep only genre and lyrics
df = df.select("genre", "lyrics").dropna()
df = df.filter(col("lyrics") != "")
df = df.filter(col("genre") != "")

# ── Balance classes 
MAX_PER_CLASS = 1500
genre_counts = df.groupBy("genre").count().collect()

balanced = []
for row in genre_counts:
    gdf = df.filter(col("genre") == row["genre"])
    if row["count"] > MAX_PER_CLASS:
        gdf = gdf.sample(fraction=MAX_PER_CLASS / row["count"], seed=42)
    balanced.append(gdf)

df = reduce(lambda a, b: a.union(b), balanced)
df = df.repartition(8)
df.cache()

print("Genre distribution after balancing:")
df.groupBy("genre").count().orderBy("count", ascending=False).show()

label_indexer  = StringIndexer(inputCol="genre", outputCol="label", handleInvalid="keep")
fitted_indexer = label_indexer.fit(df)
labels         = fitted_indexer.labels
print(f"Labels: {labels}")

# Text Pipeline
tokenizer    = Tokenizer(inputCol="lyrics", outputCol="words")
stop_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashing_tf   = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=5000)
idf          = IDF(inputCol="raw_features", outputCol="features", minDocFreq=2)

# Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    regParam=0.1,
    elasticNetParam=0.0,
    family="multinomial"
)

converter = IndexToString(
    inputCol="prediction",
    outputCol="predicted_genre",
    labels=labels
)

pipeline = Pipeline(stages=[
    fitted_indexer,
    tokenizer,
    stop_remover,
    hashing_tf,
    idf,
    lr,
    converter
])

# Training and testing data split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count()} | Test: {test_df.count()}")

model = pipeline.fit(train_df)

predictions = model.transform(test_df)

acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
).evaluate(predictions)
f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
).evaluate(predictions)

print(f"\nAccuracy : {acc*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

print("\nPer-genre results:")
predictions.groupBy("genre", "predicted_genre") \
    .count() \
    .orderBy("genre", "count", ascending=[True, False]) \
    .show(60)

model.write().overwrite().save(MODEL_PATH)

with open("data/labels.txt", "w") as f:
    f.write("\n".join(labels))

spark.stop()
