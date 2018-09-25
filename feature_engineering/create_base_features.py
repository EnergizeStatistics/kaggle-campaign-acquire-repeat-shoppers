
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
# df_transactions = spark.read.load(data_spark_dir+"transactions.parquet")
reduced_transactions = spark.read.load(data_spark_dir+"reduced_transactions.parquet")
df_train_history = spark.read.load(data_spark_dir+"train_history.parquet")
df_offers = spark.read.load(data_spark_dir+"offers.parquet")
df_test_history = spark.read.load(data_spark_dir+"test_history.parquet")

reduced_transactions.createOrReplaceTempView("reduced_transactions")
df_train_history.createOrReplaceTempView("train_history")
df_offers.createOrReplaceTempView("offers")
df_test_history.createOrReplaceTempView("test_history")
# df_transactions.createOrReplaceTempView("transactions")

# merge category/company/brand/offer value information into train history 
offer_train_history = spark.sql("\
    select a.*, b.category, b.company, b.offervalue, b.brand \
    from train_history a \
    left join offers b \
        on a.offer = b.offer \
    ")
                                
offer_train_history.createOrReplaceTempView("offer_train_history")

# merge category/company/brand/offer value information into test history 
offer_test_history = spark.sql("\
    select a.*, b.category, b.company, b.offervalue, b.brand \
    from test_history a \
    left join offers b \
        on a.offer = b.offer \
    ")
                                
offer_test_history.createOrReplaceTempView("offer_test_history")

# for each category/company/brand, calcuate the total quantity and total $ amount the customer has purchased
# during the time window
# if the customer has never bought from the category/company/brand, this information is captured too
def summarize_purchases_over(over_field, num_days_back, history_data):
    query = ('SELECT distinct id, offer, '
                 'sum(purchasequantity) as has_bought_{0}_q_{1}, '
                 'sum(purchaseamount) as has_bought_{0}_a_{1} '
             'FROM last '
             'WHERE {0} = offer_{0} '
             'GROUP BY id, offer').format(over_field, num_days_back)
    has_bought = spark.sql(query)
    # note this dataframe above does not contain customers that has never bought from their offer company
    del query
    has_bought.createOrReplaceTempView("has_bought")
    # add "never bought"
    query = ("SELECT o.*, CASE WHEN h.id IS NULL THEN 0 ELSE h.has_bought_{0}_q_{1} END AS has_bought_{0}_q_{1}, "
                 "CASE WHEN h.id IS NULL THEN 1 ELSE 0 END AS never_bought_{0}, "
                 "CASE WHEN h.id IS NULL THEN 0.0 ELSE h.has_bought_{0}_a_{1} END AS has_bought_{0}_a_{1} "
             "FROM {2} o "
                  "LEFT JOIN has_bought h "
             "ON o.id = h.id").format(over_field, num_days_back, history_data)
    has_bought_full = spark.sql(query)
    
    spark.catalog.dropTempView("has_bought")
    
    del query, has_bought
    
    return has_bought_full

# put category, company, and brand together
def extract_history(history_data, num_days_back):
    # only keep the transactions that at least have one thing (category/company/brand)
    # in common with a customer's offer
    query = ('SELECT o.id, o.offer, t.category, t.company, t.brand, '
                'o.category as offer_category, o.company as offer_company, o.brand as offer_brand, '
                't.purchasequantity, t.purchaseamount, t.date, o.offerdate '
             'FROM reduced_transactions t '
                 'INNER JOIN {} o '
                 'ON t.id = o.id '
                     'AND t.date BETWEEN date_sub(o.offerdate, {}) AND o.offerdate')\
            .format(history_data, num_days_back)
    last = spark.sql(query)
    del query
    last.createOrReplaceTempView("last")
    
    # summarize purchase history over company
    has_bought_company_full = summarize_purchases_over('company', num_days_back, history_data)
    has_bought_company_full.createOrReplaceTempView("has_bought_company_full")
    
    # summarize purchase history over category
    has_bought_category_full = summarize_purchases_over('category', num_days_back, history_data)
    has_bought_category_full.createOrReplaceTempView("has_bought_category_full")
    
    # summarize purchase history over brand
    has_bought_brand_full = summarize_purchases_over('brand', num_days_back, history_data)
    has_bought_brand_full.createOrReplaceTempView("has_bought_brand_full")
    
    # all ought to have same number of rows as trainHistory or testHistory
    assert((has_bought_company_full.count() == has_bought_category_full.count() \
            == has_bought_brand_full.count() == {}.count()).format(history_data))
    
    query = ("WITH company_category_cmb AS "
         "(SELECT co.*, ca.has_bought_category_q_{0}, "
         "ca.never_bought_category, ca.has_bought_category_a_{0} "
         "FROM has_bought_company_full co "
         "INNER JOIN has_bought_category_full ca "
         "ON (co.id = ca.id) AND (co.offer = ca.offer)) "
         "SELECT coca.*, br.has_bought_brand_q_{0}, "
         "br.never_bought_brand, br.has_bought_brand_a_{0} "
         "FROM company_category_cmb coca "
         "INNER JOIN has_bought_brand_full br "
         "ON (coca.id = br.id) AND (coca.offer = br.offer)").format(num_days_back)
    has_bought_all = spark.sql(query)
    del query, has_bought_company_full, has_bought_category_full, has_bought_brand_full 
    
    gc.collect()
    return has_bought_all

# get the dataframes
has_bought_train_30 = extract_history("offer_train_history", 30)
has_bought_train_60 = extract_history("offer_train_history", 60)
has_bought_train_90 = extract_history("offer_train_history", 90)
has_bought_train_180 = extract_history("offer_train_history", 180)

# write persistent tables to disk; do not partition by id
has_bought_train_30.write.format("parquet").save(data_spark_dir+"has_bought_train_30.parquet")
has_bought_train_60.write.format("parquet").save(data_spark_dir+"has_bought_train_60.parquet")
has_bought_train_90.write.format("parquet").save(data_spark_dir+"has_bought_train_90.parquet")
has_bought_train_180.write.format("parquet").save(data_spark_dir+"has_bought_train_180.parquet")

# get the dataframes
has_bought_test_30 = extract_history("offer_test_history", 30)
has_bought_test_60 = extract_history("offer_test_history", 60)
has_bought_test_90 = extract_history("offer_test_history", 90)
has_bought_test_180 = extract_history("offer_test_history", 180)

# write persistent tables to disk; do not partition by id
has_bought_test_30.write.format("parquet").save(data_spark_dir+"has_bought_test_30.parquet")
has_bought_test_60.write.format("parquet").save(data_spark_dir+"has_bought_test_60.parquet")
has_bought_test_90.write.format("parquet").save(data_spark_dir+"has_bought_test_90.parquet")
has_bought_test_180.write.format("parquet").save(data_spark_dir+"has_bought_test_180.parquet")

spark.stop()
