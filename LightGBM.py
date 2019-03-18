import lightgbm as lgb

### initial parameter list :
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
    'verbosity': 1
        
    }
