## load packages ##
import gc
import time
import pyspark

## specify data path ##
data_dir = '/home/lee/Documents/DatasetsForGitHub/kaggle_acquire_valued_shoppers_challenge/'
data_spark_dir = data_dir + 'spark-warehouse/'

## load data ##
spark = pyspark.sql.SparkSession.builder.appName('kaggle_acquire_shopper').getOrCreate()

transactions = spark.read.load(data_spark_dir+"transactions.parquet")
transactions.createOrReplaceTempView("transactions")

offers = spark.read.load(data_spark_dir+"offers.parquet")
offers.createOrReplaceTempView("offers")

train_history = spark.read.load(data_spark_dir+"train_history.parquet")
train_history.createOrReplaceTempView("train_history")

test_history = spark.read.load(data_spark_dir+"test_history.parquet")
test_history.createOrReplaceTempView("test_history")

# with open(data_dir+'productids.pkl', 'rb') as f:
#     productids = pickle.load(f)

def generate_features(history_data):
    # user-offer level, add offer details to train/test history
    query = ("SELECT h.*, o.category, o.company, o.brand, "
             "CONCAT(o.category, ' ', o.company, ' ', o.brand) as productid "
             "FROM {} h LEFT JOIN offers o "
             "ON h.offer = o.offer").format(history_data)
    history_prodid = spark.sql(query)
    history_prodid.createOrReplaceTempView("history_prodid")
    del query
    
    # subqueries:
    # limit30: cohort (train or test) users, transactions within 30 days prior to offer over all products, transaction level
    # totalspend: cohort users, summarize all transactions within 30 days, user level
    # prodidspend: cohort users, only product in offer, user-offer level
    # cmb_prodid_total: combine totalspend and prodidspend, user-offer level
    # final output: some cohort users may not make any purchases within 30 days, they are not in limit30,
    # add these users to the dataframe and set their spending/visits to 0, user-offer level, same # of records as 
    # train/test history
    query = ("WITH limit30 AS (SELECT t.*, h.offerdate, h.offer, h.productid, "
         "DATEDIFF(h.offerdate, t.date) AS date_diff "
         "FROM transactions t "
         "INNER JOIN history_prodid h "
         "ON t.id = h.id "
         "AND t.date BETWEEN date_sub(h.offerdate, 30) AND h.offerdate), "
         "totalspend AS (SELECT id, SUM(purchaseamount) AS total_spend_30, "
         "COUNT(DISTINCT date) AS visits_30 "
         "FROM limit30 "
         "GROUP BY id), "
         "prodidspend AS (SELECT l.id, l.category, l.company, l.brand, "
            "SUM(l.purchaseamount) AS prodid_spend_30, "
            "SUM(l.purchasequantity) AS prodid_count_30 "
         "FROM limit30 l INNER JOIN history_prodid o "
         "ON (l.id = o.id) "
         "AND (CONCAT(l.category, ' ', l.company, ' ', l.brand) = o.productid) "
         "GROUP BY l.id, l.category, l.company, l.brand), "
         "cmb_prodid_total AS (SELECT t.id, p.category, p.company, p.brand, "
         "CONCAT(p.category, ' ', p.company, ' ', p.brand) AS productid, p.prodid_spend_30, p.prodid_count_30, "
         "t.total_spend_30, t.visits_30 "
         "FROM totalspend t LEFT JOIN prodidspend p "
         "ON p.id = t.id) "
         "SELECT l.id, l.category, l.company, l.brand, "
             "COALESCE(r.prodid_spend_30, 0) AS prodid_spend_30, "
             "COALESCE(r.prodid_count_30, 0) AS prodid_count_30, "
             "COALESCE(r.total_spend_30, 0) AS total_spend_30, "
             "COALESCE(r.visits_30, 0) AS visits_30 "
         "FROM history_prodid l LEFT JOIN cmb_prodid_total r "
         "ON l.id = r.id "
            )
    user_features = spark.sql(query)
    
    assert(user_features.count() == history_prodid.count())
    
    spark.catalog.dropTempView("history_prodid")
    del query, history_prodid
    
    return user_features

user_features_train = generate_features('train_history')
user_features_test = generate_features('test_history')

# write persistent table to disk
user_features_train.write.format("parquet").save(data_spark_dir+"user_features_train.parquet")
user_features_test.write.format("parquet").save(data_spark_dir+"user_features_test.parquet")

# dffftest.filter(dffftest.offerdate.isNotNull()).show(5)

spark.stop()