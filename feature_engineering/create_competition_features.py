## load packages ##
import gc
import time
import pandas as pd
import pyspark

## specify data path ##
data_dir = '/home/lee/Documents/DatasetsForGitHub/kaggle_acquire_valued_shoppers_challenge/'
data_spark_dir = data_dir + 'spark-warehouse/'

loc_offers = data_dir + "offers.csv"

## filter transactions ##
offers = pd.read_csv(loc_offers)
# all categories
offers_cat = tuple(offers['category'].unique())
# all companies
offers_co = tuple(offers['company'].unique())
# all brands
offers_br = tuple(offers['brand'].unique())

del offers

## load data ##
spark = pyspark.sql.SparkSession.builder.appName('kaggle_acquire_shopper').getOrCreate()



reduced_transactions = spark.read.load(data_spark_dir+"reduced_transactions.parquet")

reduced_transactions.createOrReplaceTempView("reduced_transactions")

# for each category that is associated with an offer,
# 1) count the # of products (identified by unique category+company+brand combination) in that category
# 2) find the total $ amount spent by all customers
query = (
"WITH producespend AS (SELECT category, company, brand, "
    "SUM(purchaseamount) AS productspend_in_cat "
    "FROM reduced_transactions "
    "WHERE category IN {} "
    "GROUP BY category, company, brand) "
    "SELECT category, COUNT(DISTINCT company, brand) AS competing_products_in_cat, " # this is total products in category
    "MAX(productspend_in_cat)/SUM(productspend_in_cat) AS marketshare_dominant_prod_in_cat "
    "FROM producespend "
    "GROUP BY category").format(offers_cat)
competition_features = spark.sql(query)
del query

competition_features.write.format("parquet").save(data_spark_dir+"competition_features.parquet")

spark.stop()

