# Marketing Campaign Prediction #

## Description ##
This repository hosts my code to [Kaggle Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/). The task was to identify shoppers that became loyal to a product after a campaign, in which a larger number of shoppers were offered a coupon to the product. In the data each customer receives only one coupon (i.e., incentive to one product, effective on one date). Different customers may receive the same coupon or different coupons. For each customer, shopping history prior to receiving the coupon that contains all items purchased (not just items related to the coupon) is provided. The training dataset includes post-coupon behavior - the number of times the customer made a repeat purchase and a boolean value that equals to repeat purchase > 0. 

## Method ##
The shopping history dataset is a 20 GB plain text file; this large size poses the first difficulty of the analysis. As a result I chose `Spark` for data manipulation. 

Feature eniginneering, as always, is the greatest challenge. From shopping history (the "transactions" dataset) I created these features:
* whether the customer has bought the company/category/brand in the last 30/60/90/180 days (note that a unique combination of company, category, and brand identifies a unique product; a coupon is for a unique product)
* in the last 30/60/90/180 days, how much they have spent within each company/category/brand, and how many units of the product/category/brand they have bought
* for each customer, number of total shopping trips and total money spent in the last 30 days
* marketshare of a product in the category it belongs to, and the marketshare of the dominatirng product in that category
* number of different products in a category

I then built a cassification model using these features. `PySpark ML`, compared to `scikit-learn`, has limited algorithms . I compared the performances of logistic regression, random forest classifer, and gradient boosting classifer and eventually chose gradient boosting classifer. A grid search of hyperparameters was performed with 6-fold cross-vlidation. 


## Usage ##
This analysis requires Spark 2.3.1 and Python 3.6.5. 

First download all the data from the Challenge's page. Then run these scripts in the "feature_engineering"to create features 

the folder "data"
create some helper data:
create_dept_category_map.py
create_seasonal_cat.py
create_user_dates.py
create_base_features.py
create features by running these scripts. Note that you need to create features for both training and test sets. To switch between running for training and test sets, set the python variable "testset" in the beginning of each script to False or True respectively.
create_product_features1.py
create_user_features1.py
create_seasonal_features.py
create_product_cheapness_feature.py
create_rebuy_probability_categories.py
create_rebuy_probability_products.py
create_competition_features.py
create_new_product_features.py
create_user_first_transaction_feature.py
create_negative_features.py
merge_features.py
set positions for installation of xgboost, vw, sofia-ml, data-folder in top of python script generate_submission.py via variables xgboost_python_path, vowpalwabbit_path, sofiaml_path, and data_path
run generate_submission.py to create the final submission