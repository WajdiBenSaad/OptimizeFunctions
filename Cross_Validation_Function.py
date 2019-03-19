Cross_Validation_split = model_selection.ShuffleSplit(n_splits = 8, test_size = .7, train_size = .3, random_state = 43984 ) 

Algo_columns = ['Algo Name','Algo Parameters','Algo Train Acc Mean', 'Algo Test Acc Mean', 'Algo Test Acc 3*Std' ,'Algo Time']
Algo_compare = pd.DataFrame(columns = Algo_columns)
Algo_predict = train_df['target'].copy()

row_index = 0
for alg in Algos:

  
    Algo_name = alg.__class__.__name__
    Algo_compare.loc[row_index, 'Algo Name'] = Algo_name
    Algo_compare.loc[row_index, 'Algo Parameters'] = str(alg.get_params())
   
    cv_results = model_selection.cross_validate(alg,train_df[features], train_df['target'].values.ravel(), cv  = Cross_Validation_split, n_jobs=-1)
    Algo_compare.loc[row_index, 'Algo Time'] = cv_results['fit_time'].mean()
    Algo_compare.loc[row_index, 'Algo Train Acc Mean'] = cv_results['train_score'].mean()
    Algo_compare.loc[row_index, 'Algo Test Acc Mean'] = cv_results['test_score'].mean()   

    Algo_compare.loc[row_index, 'Algo Test Acc 3*Sdt'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    alg.fit(train_df[features], train_df['target'].values.ravel() )
    Algo_predict[Algo_name] = alg.predict(train_df[features])
    
row_index+=1
