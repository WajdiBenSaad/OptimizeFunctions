import lightgbm as lgb
## Data :
y_train  
y_test  
X_train  
X_test

# Dataset for lightgbm 
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


### initial parameter list for a classification problem:

  param = {
    'bagging_freq': 4,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.04,
    'learning_rate': 0.001,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
            }

## Model Definition:

 lgb_model = lgb.train(param,lgb_train,num_boost_round=20,valid_sets=lgb_eval,early_stopping_rounds=5)

  
## Model predictions

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

## Model evalution
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
