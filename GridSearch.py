###Getting the libraries

import os
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV
import lightgbm as lgm

### Implementing the gridsearch function to the LightGBM algorithm:
###---------------------------------------------------------------###
  params = {
        "objective" : "binary",
        "metric" : "auc",
        "max_depth" : -1,
        "num_leaves" : 8,
        "min_data_in_leaf" : 25,
        "learning_rate" : 0.006,
        "bagging_fraction" : 0.2,
        "feature_fraction" : 0.4,
        "bagging_freq" : 1,
        "lambda_l1" : 5,
        "lambda_l2" : 5,
        "verbosity" : 1,
        "max_bin": 512,
        "num_threads" : 6
    }

  gridParams = {
    'learning_rate': [0.005],
    'n_estimators': [500,800,1200],
    'num_leaves': [8,4,2],
    'colsample_bytree' : [0.7,0.07],
    'reg_alpha' : [1,1.2,7],
    'reg_lambda' : [5,5.5,0.6],
    'subsample' : [0.7,0.07],
    'objective' : ['binary'],
    'boosting_type' : ['gbdt'],
}
#    'boosting_type' : ['gbdt'],
#    'objective' : ['binary'],
#    'random_state' : [501], # Updated from 'seed'
#    'colsample_bytree' : [0.65, 0.66],
#    'subsample' : [0.7,0.75],
#    'reg_alpha' : [1,1.2],
#    'reg_lambda' : [1,1.2,1.4],
#    }
