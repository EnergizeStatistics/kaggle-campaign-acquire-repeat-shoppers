# Marketing Campaign Prediction #

## Description ##
This repository hosts my code to [Kaggle Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/). 

The task is to identify shoppers that become loyal to a product after a campaign, in which many shoppers are offered a coupon to the product. In the data each customer receives only one coupon, where one coupon is applicable to only one unique product (identified by a unique combination of company, category, and brand). Different customers may receive the same coupon or different coupons. For each customer, shopping history prior to receiving the coupon that contains all items s/he purchased (not just items related to the coupon) is provided. The training dataset includes post-coupon behavior - the number of times the customer makes a repeat purchase and a Boolean value that equals to repeat purchase > 0. 

## Method ##
The shopping history dataset is a 20 GB plain text file; this enormous size poses the first difficulty of the analysis. As a result, I chose `Spark` for data manipulation. 

Feature engineering, as always, is the greatest challenge. From shopping history (the "transactions" data file) I created these features:
* whether the customer has bought the product's company/category/brand related in the last 30/60/90/180 days
* in the last 30/60/90/180 days, the total spending of the customer in each company/category/brand, and the numbers of units of the product/category/brand purchased
* for each customer, number of total shopping trips and total money spent in the last 30 days
* market share of a product in the category it belongs to, and the market share of the dominating product in that category
* number of different products in a category

I then built a classification model using these features. `PySpark ML`, compared to `scikit-learn`, has limited algorithms. I compared the performances of logistic regression, random forest classifier, and gradient boosting classifier and eventually chose gradient boosting classifier. A grid search of hyperparameters was performed with 6-fold cross-validation. 

## Usage ##
This analysis requires `Spark 2.3.1` and `Python 3.6.5`. 

First download all the data from the Challenge's page.

Run load_raw_to_spark.py and then reduce_transactions.py. Recall that the original shopping history covers all purchased items; reduce_transactions.py selects the shopping records of the company/category/brand related to the customer's coupon offer. 

Then run the scripts in the "feature_engineering" folder to create features; the order of these scripts does not matter and one can skip any of them if so chooses: 
* create_base_features.py
* create_product_features.py
* create_user_features.py
* create_competition_features.py.

Then run merge_features.py. If any of the earlier feature construction scripts are skipped, the corresponding parquet files need to be removed from this script. 

Finally run train_predict.py. This script generates a csv submission file. 

## Potential Improvement ##
Opportunties for improvement lie beyond `Spark`. If we took the data out of `Spark` and into `pandas`, we would be able to utilize the rich selection of algorithms of `sci-kit learn` at the cost of losing the distributed implementation in `Spark ML`. This approach is viable for datasets that fit in memory. We could look at feature importance and remove the non-important features in order to reduce dataset sizes. Model stacking would also be a natural step after training with various algorithms. 