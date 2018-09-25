## load packages ##
import time
import gc
import pandas as pd
import pyspark

## specify data path ##
data_dir = '/home/lee/Documents/DatasetsForGitHub/kaggle_acquire_valued_shoppers_challenge/'

loc_offers = data_dir + "offers.csv"
loc_transactions = data_dir + "transactions.csv"
loc_train = data_dir + "trainHistory.csv"
loc_test = data_dir + "testHistory.csv"

## load data ##
spark = pyspark.sql.SparkSession.builder.appName('kaggle_acquire_shopper').getOrCreate()

def load_to_spark(loc):
    return spark.read.options(header=True, inferSchema=True).csv(loc)

start_time = time.time()
df_transactions = load_to_spark(loc_transactions)
print('loading transactions into Spark takes {0} seconds'.format((time.time() - start_time)))
del start_time
gc.collect()


start_time = time.time()
df_train_history = load_to_spark(loc_train)
print('loading train history into Spark takes {0} seconds'.format((time.time() - start_time)))
del start_time

start_time = time.time()
df_test_history = load_to_spark(loc_test)
print('loading test history into Spark takes {0} seconds'.format((time.time() - start_time)))
del start_time

start_time = time.time()
df_offers = load_to_spark(loc_offers)
print('loading offers into Spark takes {0} seconds'.format((time.time() - start_time)))
del start_time

# preview
def preview_df(df_nm):
    print('Preview Spark DataFrame:\n {}'.format(locals()))
    df_nm.printSchema()
    df_nm.show(10)


# comment out when pipelining
# preview_df(df_transactions)
# preview_df(df_train_history)
# preview_df(df_offers)

# category-department mapping
cat_dept_map_spark = df_transactions.groupby(['dept','category']).count().collect()
cat_dept_map = pd.DataFrame([lambda x x.asDict() for x in cat_dept_map_spark])

## write persistent tables ##
# df_transactions.write.format("parquet").partitionBy("id").save(data_dir+"spark-warehouse/transactions.parquet")
# partitioning by id is too demanding as there are many ids; loading parititioned parquet gives out-of-memory error
df_transactions.write.format("parquet").save(data_dir+"spark-warehouse/transactions.parquet")
df_train_history.write.format("parquet").save(data_dir+"spark-warehouse/train_history.parquet")
df_test_history.write.format("parquet").save(data_dir+"spark-warehouse/test_history.parquet")
df_offers.write.format("parquet").save(data_dir+"spark-warehouse/offers.parquet")

cat_dept_map.to_pickle(data_dir+"cat_dept_map.pkl")

spark.stop()