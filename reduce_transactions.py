## load packages ##
import time
import pandas as pd
import gc
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


spark = pyspark.sql.SparkSession.builder.appName('kaggle_acquire_shopper').getOrCreate()

df_transactions = spark.read.load(data_spark_dir+"transactions.parquet")
df_transactions.createOrReplaceTempView("transactions")

query = ("SELECT * "
    "FROM transactions "
    "WHERE (category IN {}) "
         "OR (company IN {}) "
         "OR (brand IN {})").format(offers_cat, offers_co, offers_br)

reduced_transactions = spark.sql(query)

# partitioning by id is too demanding as there are many ids; loading parititioned parquet gives out-of-memory error
# reduced_transactions.write.format("parquet").partitionBy("id").save(data_spark_dir+"reduced_transactions.parquet")
reduced_transactions.write.format("parquet").save(data_spark_dir+"reduced_transactions.parquet")

spark.stop()

# import csv
# loc_transactions = data_dir + "transactions.csv" # will be created
# loc_reduced = data_dir + "transactions_reduced.csv" 
# def filter_transactions():
#     # do not open file as binary, open as text
#     with open(loc_transactions, "rt") as infile:
    
#         datareader = csv.reader(infile)
#         # the header row
#         yield next(datareader)  
        
#         # the data rows that meet the criteria: 
#         yield from filter(lambda row: (row[3] in offers_cat) | (row[4] in offers_co), datareader)
        
#         return

# ### execute generator ###
# mygenerator = filter_transactions()

# start_time = time.time()
# # write generator to a file
# with open(loc_reduced, "wt") as outfile:
#     for i in mygenerator:
#         outfile.write(",".join(i))
#         outfile.write("\n")
# print('writing reduced transactions to csv takes {} seconds'.format(time.time() - start_time))

# del start_time, offers, offers_cat, offers_co, mygenerator 
# gc.collect()

