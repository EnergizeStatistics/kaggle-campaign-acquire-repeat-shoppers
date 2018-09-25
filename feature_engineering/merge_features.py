## load packages ##

import gc
import time
import pyspark

## specify data path ## 

data_dir = '/home/lee/Documents/DatasetsForGitHub/kaggle_acquire_valued_shoppers_challenge/'
data_spark_dir = data_dir + 'spark-warehouse/'

# entry point
spark = pyspark.sql.SparkSession.builder.appName('kaggle_acquire_shopper').getOrCreate()

# read persistent tables into Spark session
# transactions = spark.read.load(data_spark_dir+"transactions.parquet")
# transactions.createOrReplaceTempView("transactions")

reduced_transactions = spark.read.load(data_spark_dir+"reduced_transactions.parquet")
reduced_transactions.createOrReplaceTempView("reduced_transactions")

offers = spark.read.load(data_spark_dir+"offers.parquet")
offers.createOrReplaceTempView("offers")

train_history = spark.read.load(data_spark_dir+"train_history.parquet")
train_history.createOrReplaceTempView("train_history")

test_history = spark.read.load(data_spark_dir+"test_history.parquet")
test_history.createOrReplaceTempView("test_history")
# df_transactions.createOrReplaceTempView("transactions")

# purchase history features
has_bought_train_30 = spark.read.load(data_spark_dir+"has_bought_train_30.parquet")
has_bought_train_30.createOrReplaceTempView("has_bought_train_30")
has_bought_train_60 = spark.read.load(data_spark_dir+"has_bought_train_60.parquet")
has_bought_train_60.createOrReplaceTempView("has_bought_train_60")
has_bought_train_90 = spark.read.load(data_spark_dir+"has_bought_train_90.parquet")
has_bought_train_90.createOrReplaceTempView("has_bought_train_90")
has_bought_train_180 = spark.read.load(data_spark_dir+"has_bought_train_180.parquet")
has_bought_train_180.createOrReplaceTempView("has_bought_train_180")

has_bought_test_30 = spark.read.load(data_spark_dir+"has_bought_test_30.parquet")
has_bought_test_30.createOrReplaceTempView("has_bought_test_30")
has_bought_test_60 = spark.read.load(data_spark_dir+"has_bought_test_60.parquet")
has_bought_test_60.createOrReplaceTempView("has_bought_test_60")
has_bought_test_90 = spark.read.load(data_spark_dir+"has_bought_test_90.parquet")
has_bought_test_90.createOrReplaceTempView("has_bought_test_90")
has_bought_test_180 = spark.read.load(data_spark_dir+"has_bought_test_180.parquet")
has_bought_test_180.createOrReplaceTempView("has_bought_test_180")

# user features
user_features_train = spark.read.load(data_spark_dir+"user_features_train.parquet")
user_features_train.createOrReplaceTempView("user_features_train")

user_features_test = spark.read.load(data_spark_dir+"user_features_test.parquet")
user_features_test.createOrReplaceTempView("user_features_test")

# product features
product_features = spark.read.load(data_spark_dir+"product_features.parquet")
product_features.createOrReplaceTempView("product_features")

competition_features = spark.read.load(data_spark_dir+"competition_features.parquet")
competition_features.createOrReplaceTempView("competition_features")

def merge_all_features(history_dataset):

    query = ("WITH add_offer_details AS ( "
             "SELECT h.*, u.category, u.company, u.brand, "
             "CONCAT(u.category, ' ', u.company, ' ', u.brand) AS productid "
             "FROM {0}_history h LEFT JOIN offers u "
             "ON h.offer = u.offer), "
             "add_user_features AS ( "
             "SELECT h.*, "
             "u.prodid_spend_30, u.prodid_count_30, u.total_spend_30, u.visits_30 "
             "FROM add_offer_details h LEFT JOIN user_features_{0} u "
             "ON h.id = u.id), "
             "add_product_features AS ( "
             "SELECT h.*, p.marketshare_in_cat "
             "FROM add_user_features h LEFT JOIN product_features p "
             "ON h.productid = p.productid), "
             "add_has_bought_30_features AS ( "
             "SELECT h.*, b.has_bought_company_q_30, "
             "b.never_bought_company, b.has_bought_company_a_30, "
             "b.has_bought_category_q_30, b.never_bought_category, "
             "b.has_bought_category_a_30, b.has_bought_brand_q_30, "
             "b.never_bought_brand, b.has_bought_brand_a_30 "
             "FROM add_product_features h LEFT JOIN has_bought_{0}_30 b "
             "ON h.id = b.id), "
             "add_has_bought_60_features AS ( "
             "SELECT h.*, b.has_bought_company_q_60, "
             "b.has_bought_company_a_60, "
             "b.has_bought_category_q_60, "
             "b.has_bought_category_a_60, b.has_bought_brand_q_60, "
             "b.has_bought_brand_a_60 "
             "FROM add_has_bought_30_features h LEFT JOIN has_bought_{0}_60 b "
             "ON h.id = b.id), "
             "add_has_bought_90_features AS ( "
             "SELECT h.*, b.has_bought_company_q_90, "
             "b.has_bought_company_a_90, "
             "b.has_bought_category_q_90, "
             "b.has_bought_category_a_90, b.has_bought_brand_q_90, "
             "b.has_bought_brand_a_90 "
             "FROM add_has_bought_60_features h LEFT JOIN has_bought_{0}_90 b "
             "ON h.id = b.id), "
             "add_has_bought_180_features AS ( "
             "SELECT h.*, b.has_bought_company_q_180, "
             "b.has_bought_company_a_180, "
             "b.has_bought_category_q_180, "
             "b.has_bought_category_a_180, b.has_bought_brand_q_180, "
             "b.has_bought_brand_a_180 "
             "FROM add_has_bought_90_features h LEFT JOIN has_bought_{0}_180 b "
             "ON h.id = b.id) "
             "SELECT h.*, b.competing_products_in_cat, "
             "b.marketshare_dominant_prod_in_cat "
             "FROM add_has_bought_180_features h LEFT JOIN competition_features b "
             "ON h.category = b.category "        
            ).format(history_dataset)


    all_features = spark.sql(query)
    del query
    
    return all_features

all_features_train = merge_all_features('train')

all_features_test = merge_all_features('test')


# write persistent tables to disk; do not partition by id
all_features_train.write.format("parquet").save(data_spark_dir+"all_features_train.parquet")
all_features_test.write.format("parquet").save(data_spark_dir+"all_features_test.parquet")
