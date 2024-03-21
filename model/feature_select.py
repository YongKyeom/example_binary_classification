import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier

def fnFeatSelect_RandVar(
    df_x, 
    df_y, 
    x_var, 
    core_cnt, 
    rand_num = 10, 
    threshold = 0.00, 
    seed = 1000
):
    """
       To select feature with random variables
       1. Add random variables(random sampling)
       2. Fit random forest
       3. Select feature which has more feature importance than random variables

           Args:
               df: Total data
               x_var: feature list of ml_model
               y_var: target variable's name
               rand_num: cnt of adding variables
               threshold: threshold of feature importance

           Returns:
               feature list(selected)
    """
    
    try:
        df_rand     = df_x.copy()
        random_cols = []
        np.random.seed(seed)

        for i0 in range(1, rand_num + 1):
            ## name of random feature
            random_col = '__random_{}__'.format(i0)

            ## add random feature
            df_rand[random_col] = np.random.rand(df_rand.shape[0])

            ## add random feature name
            random_cols.append(random_col)

        ## Fit RF
        model_rf = RandomForestClassifier(n_estimators = 500, n_jobs = core_cnt, random_state = seed)
        model_rf.fit(df_rand[x_var + random_cols], df_y)

        ## Feature importance
        feat_imp_df = pd.DataFrame({
            'feature_name': x_var + random_cols,
            'feature_importance': model_rf.feature_importances_
            }
        )

        ## Sort feature importance
        feat_imp_df = feat_imp_df.sort_values('feature_importance', ascending = False).reset_index(drop = True)

        ## Importance of random features
        imp_random    = feat_imp_df.loc[feat_imp_df.feature_name.isin(random_cols), 'feature_importance'].values
        imp_threshold = max(np.percentile(imp_random, 50), threshold)

        ## Filter with imp_threshold
        feat_imp_filter = feat_imp_df[feat_imp_df['feature_importance'] > imp_threshold]

        ## Selet feature
        feat_select = list(set(feat_imp_filter.feature_name) - set(random_cols))
        feat_select.sort()

    except Exception as e:
        print('Error in fnFeatSelect_RandVar')
        print(e)
        
        raise Exception('Check error')

    return feat_select