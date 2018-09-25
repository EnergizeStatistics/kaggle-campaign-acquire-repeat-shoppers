## load packages ##
import gc, time, pickle, os
import pyspark
import pandas as pd

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

## specify data path ## 
data_dir = '/home/lee/Documents/DatasetsForGitHub/kaggle_acquire_valued_shoppers_challenge/'
data_spark_dir = data_dir + 'spark-warehouse/'

# entry point
spark = pyspark.sql.SparkSession.builder.appName('kaggle_acquire_shopper').getOrCreate()

# read data
all_features_train = spark.read.load(data_spark_dir+"all_features_train.parquet")
all_features_test = spark.read.load(data_spark_dir+"all_features_test.parquet")

# prepare for training
X_feature_names = list(set(all_features_train.schema.names) - set(('id',
 'chain',
 'offer',
 'market',
 'repeattrips',
 'repeater',
 'offerdate',
 'category',
 'company',
 'brand',
 'productid')))

# data transformer
assembler_features = VectorAssembler(inputCols=X_feature_names, outputCol='features')
labelIndexer = StringIndexer(inputCol='repeater', outputCol="label")

# classificiation model
clf_gbt = GBTClassifier(featuresCol='features', labelCol="label", seed=0)

pipeline = Pipeline(stages=[assembler_features, labelIndexer, clf_gbt])

# no parameter search
# paramGrid = ParamGridBuilder().build()

# with parameter search
paramGrid = (ParamGridBuilder()
             .addGrid(clf_gbt.maxDepth, [2, 4, 6])
             .addGrid(clf_gbt.maxBins, [20, 60])
             .addGrid(clf_gbt.maxIter, [10, 20])
             .build())

# 6-fold cross validation
crossval = CrossValidator(
    estimator=pipeline, \
    estimatorParamMaps=paramGrid, \
    evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC"), \
    numFolds=6)

cvModel = crossval.fit(all_features_train)

# make predictions on training. cvModel uses the best model found
predicted = cvModel.transform(all_features_train)

# save model
cvModel.save(os.path.join(data_dir, 'model_GBTClassifer'))

evaluator = BinaryClassificationEvaluator()
print("Training ROC with 6-fold cross validation: {:.6f}"      .format(evaluator.evaluate(predicted, {evaluator.metricName: "areaUnderROC"})))

# make predictions on test set
test_predicted = cvModel.transform(all_features_test)

# extract predicted probability for repeater=t
secondelement=udf(lambda v:float(v[1]), FloatType())
submit_pred_prob = test_predicted.select(secondelement('probability'))

# generate submission file
df_submit_pred_prob = submit_pred_prob.toPandas()
submission_id = pd.read_csv(data_dir+'sampleSubmission.csv')
submission_gbt = pd.concat([submission_id, df_submit_pred_prob], sort=False, axis=1).drop('repeatProbability', axis=1).rename({'<lambda>(probability)':'repeatProbability'}, axis='columns')

assert len(submission_id)==len(df_submit_pred_prob)

submission_gbt.to_csv(data_dir+'submission_gbt.csv', index=False)

