
############################
## ALL MODELS
############################
Algos = [
  
   #Xgboost
    XGBClassifier() ,
  
   #Ensemble Methods
     ensemble.AdaBoostClassifier(),
     ensemble.BaggingClassifier(),
     ensemble.ExtraTreesClassifier(),
     ensemble.GradientBoostingClassifier(),
     ensemble.RandomForestClassifier(), #Gaussian Processes
     gaussian_process.GaussianProcessClassifier(),
    
#GLM 
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier()
    linear_model.Perceptron(),
    
#Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    

  
 
    
      
    ]
