## load packages ##
import gc
import time
import pandas as pd
import pyspark
import pickle

## specify data path ##
data_dir = '/home/lee/Documents/DatasetsForGitHub/kaggle_acquire_valued_shoppers_challenge/'
data_spark_dir = data_dir + 'spark-warehouse/'

## load data ##
spark = pyspark.sql.SparkSession.builder.appName('kaggle_acquire_shopper').getOrCreate()

reduced_transactions = spark.read.load(data_spark_dir+"reduced_transactions.parquet")
reduced_transactions.createOrReplaceTempView("reduced_transactions")

df_offers = spark.read.load(data_spark_dir+"offers.parquet")
df_offers.createOrReplaceTempView("offers")

# all categories associated with an offer
query = ("SELECT DISTINCT category FROM offers")
offers_cat = tuple(spark.sql(query).toPandas()['category'].tolist())
del query

# all distinct product ids
query = ("SELECT DISTINCT CONCAT(category, ' ', company, ' ', brand) as productid FROM offers")
productids = tuple(spark.sql(query).toPandas()['productid'].tolist())
del query
# save product ids
with open(data_dir+'productids.pkl', 'wb') as f:
    pickle.dump(productids, f)
    
# marketshare_in_cat: of transactions that ever bought the product / # of transactions ever bought the category
query = (
"WITH product_transactions AS (SELECT CONCAT(category, ' ',  company, ' ', brand) as productid, "
    "category, company, brand, "
    "COUNT(*) AS mshare_product "
    "FROM reduced_transactions "
    "GROUP BY category, company, brand), "
    "category_transactions AS (SELECT category, "
    "count(*) AS category_bought "
    "FROM reduced_transactions "
    "WHERE category IN {0} "
    "GROUP BY category) "
    "SELECT p.productid, p.category, p.company, p.brand, "
    "p.mshare_product/c.category_bought AS marketshare_in_cat " 
    "FROM product_transactions p LEFT JOIN category_transactions c "
    "ON p.category = c.category "
    "WHERE p.productid IN {1}"
    ).format(offers_cat, productids)
product_features = spark.sql(query)
del query

# write persistent table to disk
product_features.write.format("parquet").save(data_spark_dir+"product_features.parquet")

spark.stop()
