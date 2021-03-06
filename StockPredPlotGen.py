import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import pyspark
from pyspark.sql import SparkSession
import os

try:
    FILE_PATH = input("Please enter the path of the source stock csv file you wish to use : ")
    FILE_NAME = FILE_PATH.split("/")[-1].split(".csv")[0]
except:
    raise Exception("Invalid File Type or Path")


spark = SparkSession.builder.master("local[1]").appName("StockPrediction").getOrCreate()

def read_file(FILE_PATH):
    df = spark.read.csv(FILE_PATH)
    if "_c7" in df.columns:
        df = df.drop("_c7")
    df = df.withColumnRenamed("_c0","Date").withColumnRenamed("_c1","Open").withColumnRenamed("_c2","High").withColumnRenamed("_c3","Low").withColumnRenamed("_c4","Close").withColumnRenamed("_c5","Adj Close").withColumnRenamed("_c6","Volume")
    df = df.where(df.Date != "Date")
    days = lambda i: i * 86400
    w = (Window.orderBy(to_timestamp(F.col("Date")).cast('long')).rangeBetween(-days(7), 0))
    w2 = Window.partitionBy().orderBy("date")
    df = df.withColumn('rolling_average', F.avg("Close").over(w))
    df = df.withColumn('is_higher_than_rollavg', F.when(F.lag(df["Close"]).over(w2) < F.col("Close"), 'yep').otherwise('nah'))
    df = df.withColumn('diffOpenClose', df["Open"] - df["Close"])
    df = df.withColumn('diffHighLow', df["High"] - df["Low"])
    return df

df = read_file(FILE_PATH)

def linearRegression3features(df):
    df = df.withColumn('Open', F.col("Open").cast('float'))
    df = df.withColumn('High', F.col("High").cast('float'))
    df = df.withColumn('Low', F.col("Low").cast('float'))
    df = df.withColumn('Close', F.col("Close").cast('float'))
    df = df.withColumn('diffOpenClose', F.col("diffOpenClose").cast('float'))
    df = df.withColumn('diffHighLow', F.col("diffHighLow").cast('float'))
    df = df.withColumn('rolling_average', F.col("rolling_average").cast('float'))
    df = df.drop("is_higher_than_rollavg")

    assembler = VectorAssembler(inputCols=["rolling_average","diffOpenClose","diffHighLow"], outputCol="features")
    vecTrainDF = assembler.transform(df)
    vecTrainDF.select("rolling_average","diffOpenClose","diffHighLow", "features", "Close").show(10)
    pipeline = Pipeline(stages=[assembler])
    pipelineModel = pipeline.fit(df)
    res = pipelineModel.transform(df)

    (trainingData, testData) = res.randomSplit([0.8, 0.2])

    lr = LinearRegression(featuresCol="features", labelCol="Close")
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)

    evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    predictions = predictions.withColumn("Error", F.col("Close") - F.col("prediction"))
    predictions.show(10)
    return rmse, predictions

def linearRegressionAlone(df):
    df = df.withColumn('Open', F.col("Open").cast('float'))
    df = df.withColumn('High', F.col("High").cast('float'))
    df = df.withColumn('Low', F.col("Low").cast('float'))
    df = df.withColumn('Close', F.col("Close").cast('float'))
    df = df.withColumn('diffOpenClose', F.col("diffOpenClose").cast('float'))
    df = df.withColumn('diffHighLow', F.col("diffHighLow").cast('float'))
    df = df.withColumn('rolling_average', F.col("rolling_average").cast('float'))
    df = df.drop("is_higher_than_rollavg")

    assembler = VectorAssembler(inputCols=["rolling_average"], outputCol="features")
    vecTrainDF = assembler.transform(df)
    vecTrainDF.select("rolling_average", "features", "Close").show(10)
    pipeline = Pipeline(stages=[assembler])
    pipelineModel = pipeline.fit(df)
    res = pipelineModel.transform(df)

    (trainingData, testData) = res.randomSplit([0.8, 0.2])

    lr = LinearRegression(featuresCol="features", labelCol="Close")
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)

    evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    predictions = predictions.withColumn("Error", F.col("Close") - F.col("prediction"))
    predictions.drop("Open").drop("High").drop("Low").drop("Adj Close").drop("features").show(10)
    return rmse, predictions

def DecisionTreeRegressorLessCategorical(df):
    df = df.withColumn('Open', F.col("Open").cast('float'))
    df = df.withColumn('High', F.col("High").cast('float'))
    df = df.withColumn('Low', F.col("Low").cast('float'))
    df = df.withColumn('Close', F.col("Close").cast('float'))
    df = df.withColumn('diffOpenClose', F.col("diffOpenClose").cast('float'))
    df = df.withColumn('diffHighLow', F.col("diffHighLow").cast('float'))
    df = df.withColumn('rolling_average', F.col("rolling_average").cast('float'))
    stages = []

    label_stringIdx = StringIndexer(inputCol='is_higher_than_rollavg', outputCol='label')
    stages += [label_stringIdx]

    assembler = VectorAssembler(inputCols=["diffOpenClose","rolling_average","diffHighLow"], outputCol="features")

    stages += [assembler]

    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    df.select('Close', 'label', 'features').show()

    (trainingData, testData) = df.randomSplit([0.8, 0.2])

    dr = DecisionTreeRegressor(labelCol="Close", featuresCol="features")

    model = dr.fit(trainingData)
    predictions = model.transform(testData)
    predictions.select("Close","label","prediction").show(10)

    evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    return rmse, predictions

def saveGraph(algo, df, name, title):
    linear3_score = algo(df)
    dtf = linear3_score[1].toPandas()
    dtf['Date'] = pd.to_datetime(dtf['Date'])
    dtf = dtf.set_index("Date")
    plt.rcParams["figure.figsize"] = (12.8,9.6)
    plt.title(title+" : Root Mean Squared Error = "+str(linear3_score[0]))
    dtf["Close"].plot()
    dtf["prediction"].plot()
    plt.legend(['Reality',"Prediction"])
    plt.savefig(name)

if __name__ == "__main__":
    os.system("mkdir %s" % (FILE_NAME))
    saveGraph(linearRegressionAlone, df, "%s/linearRegressionAlone-%s.jpg" % (FILE_NAME,FILE_NAME), "%s linearRegression on 1 features" % (FILE_NAME))
    plt.clf()
    df = read_file(FILE_PATH)
    saveGraph(linearRegression3features, df, "%s/linearRegression3features-%s.jpg" % (FILE_NAME,FILE_NAME), "%s linearRegression on 3 features" % (FILE_NAME))
    plt.clf()
    df = read_file(FILE_PATH)
    saveGraph(DecisionTreeRegressorLessCategorical, df, "%s/DecisionTreeRegressor-%s.jpg" % (FILE_NAME,FILE_NAME), "%s DecisionTreeRegressor on 4 features" % (FILE_NAME))
