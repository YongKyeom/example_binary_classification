import numpy as np
import pandas as pd
import datetime

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from hyperopt import fmin, hp, STATUS_OK, Trials, tpe

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


def fnOpt_HyperPara(
    total_data, 
    x_var, 
    y_var, 
    space, 
    lean_rate_ls, 
    ml_model = 'rf', 
    core_cnt = 3, 
    cv_num = 3, 
    max_evals = 50, 
    seed = 1000, 
    verbose = False, 
):
    """
       To optimizing hyper-parameter with Baysian optimizing

           Args:
               total_data: data frame
               x_var: x feature list(Ex. x_var = ['a', 'b', 'c'])
               y_var: target column name(Ex. y_var = 'ESTM_SOH_DIFF')
               space: Search space with baysian optimizing
               lean_rate_ls: If model in ['xgb', 'lgbm'], using to optimize learning rate
               ml_model: ['rf', 'xgb', 'lgbm']
               core_cnt: Count of cpu core
               cv_num: Count of validation in calculating error
               max_evals: Maximum trying number of baysian optimizing
               verbose: If True, print best hyper-parameter and time spending
           Returns:
               feature list(selected)
    """

    if verbose:
        st_time = datetime.datetime.now()
        print('Optimizing Hyper-Para({}) is Start'.format(ml_model.upper()))

    try:
        ## Train / Test Data
        kfold_data = KFold(n_splits = cv_num, shuffle = True, random_state = seed)

        if ml_model == 'rf':

            ## Add max_features
            space['max_features'] = hp.quniform('max_features', 3, len(x_var), 1)

            # ## Score List
            # cross_score   = []

            def objective(space):

                ## Score List
                cross_score   = []

                for i0, (train_idx, valid_index) in enumerate(kfold_data.split(total_data[x_var])):

                    ## Split Train/Test
                    train_df = total_data.iloc[train_idx]
                    valid_df = total_data.iloc[valid_index]

                    model = RandomForestClassifier(n_estimators      = 300,
                                                   max_depth         = int(space['max_depth']),
                                                   min_samples_leaf  = int(space['min_samples_leaf']),
                                                   min_samples_split = int(space['min_samples_split']),
                                                   max_features      = int(space['max_features']),
                                                   n_jobs            = core_cnt,
                                                   random_state      = seed)

                    model.fit(train_df[x_var], train_df[y_var])

                    ## Score
                    score = log_loss(valid_df[y_var], model.predict_proba(valid_df[x_var]))
                    cross_score.append(score)

                return {'loss': np.nanmean(cross_score), 'status': STATUS_OK}

        elif ml_model == 'xgb':

            # ## Score List
            # cross_score   = []

            def objective(space):

                ## Score List
                cross_score   = []

                for i0, (train_idx, valid_index) in enumerate(kfold_data.split(total_data[x_var])):

                    ## Split Train/Test
                    train_df = total_data.iloc[train_idx]
                    valid_df = total_data.iloc[valid_index]

                    model = XGBClassifier(n_estimators     = 300,
                                          max_depth        = int(max(space['max_depth'], 3)),
                                          min_child_weight = int(max(space['min_child_weight'], 1)),
                                          learning_rate    = 0.01,
                                          gamma            = space['gamma'],
                                          subsample        = min(space['subsample'], 1),
                                          colsample_bytree = min(space['colsample_bytree'], 1),
                                          n_jobs           = core_cnt,
                                          random_state     = seed)

                    model.fit(train_df[x_var],
                              train_df[y_var],
                              early_stopping_rounds = 15,
                              eval_set = [(train_df[x_var], train_df[y_var]),
                                          (valid_df[x_var], valid_df[y_var])],
                              verbose = False)

                    ## Score
                    score = log_loss(
                        valid_df[y_var], 
                        model.predict_proba(valid_df[x_var])
                    )
                    cross_score.append(score)

                return {'loss': np.nanmean(cross_score), 'status': STATUS_OK}

        elif ml_model == 'lgbm':

            # ## Score List
            # cross_score   = []

            def objective(space):

                ## Score List
                cross_score   = []

                for i0, (train_idx, valid_index) in enumerate(kfold_data.split(total_data[x_var])):

                    ## Split Train/Test
                    train_df = total_data.iloc[train_idx]
                    valid_df = total_data.iloc[valid_index]

                    model = LGBMClassifier(n_estimators     = 300,
                                           max_depth        = int(max(space['max_depth'], 3)),
                                           min_child_weight = space['min_child_weight'],
                                           num_leaves       = int(max(space['num_leaves'], 5)),
                                           learning_rate    = 0.01,
                                           subsample        = min(space['subsample'], 1),
                                           colsample_bytree = min(space['colsample_bytree'], 1),
                                           n_jobs           = core_cnt,
                                           random_state     = seed)

                    model.fit(train_df[x_var],
                              train_df[y_var],
                              eval_set = [(train_df[x_var], train_df[y_var]),
                                          (valid_df[x_var], valid_df[y_var])],
                              callbacks = [
                                    early_stopping(
                                        stopping_rounds = 15,
                                        verbose = False
                                        )
                                    ]
                              )

                    ## Score
                    score = log_loss(
                        valid_df[y_var], 
                        model.predict_proba(valid_df[x_var])
                    )
                    cross_score.append(score)

                return {'loss': np.nanmean(cross_score), 'status': STATUS_OK}

        trials = Trials()
        best_para = fmin(fn        = objective,
                         space     = space,
                         algo      = tpe.suggest,
                         max_evals = max_evals,
                         trials    = trials,
                         rstate    = np.random.Generator(np.random.PCG64(seed)))

        if ml_model == 'rf':
            for i0, v0 in best_para.items():
                best_para[i0] = int(v0)

        elif ml_model in ['xgb', 'lgbm']:
            best_para['max_depth']        = int(max(best_para['max_depth'], 1))
            best_para['subsample']        = min(round(best_para['subsample'], 2), 1)
            best_para['colsample_bytree'] = min(round(best_para['colsample_bytree'], 2), 1)
            
            if ml_model == 'xgb':
                best_para['gamma'] = round(best_para['gamma'], 2)

            if ml_model == 'lgbm':
                best_para['num_leaves'] = int(max(best_para['num_leaves'], 5))

        ## L1 & L2 Regularization
        if ml_model in ['xgb', 'lgbm']:
            
            print('Optimizing L1, L2 Regularization')

            ## Initial learning_rate
            best_para['learning_rate'] = 0.01
            ## Initial n_estimators
            best_para['n_estimators']  = 500

            for reg_type in ['reg_alpha', 'reg_lambda']:
                if reg_type == 'reg_alpha':
                    print(f'.. Start: L1({reg_type})')
                else:
                    print(f'.. Start: L2({reg_type})')
                
                ## Regularization List
                l1l2_reg_ls = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

                ## Number of Estimator List
                reg_ls        = []
                ## Score List
                score_ls      = []
                ## Number of Estimator
                nestimator_ls = []

                ## Train / Test Data
                kfold_data = KFold(n_splits = cv_num, shuffle = True, random_state = seed + 1)

                for i0, (train_idx, valid_index) in enumerate(kfold_data.split(total_data[x_var])):
                    ## Split Train/Test
                    train_df = total_data.iloc[train_idx]
                    valid_df = total_data.iloc[valid_index]

                    for reg_panalty in l1l2_reg_ls:
                        ## Add Regularization
                        best_para[reg_type]  = reg_panalty
                        reg_ls.append(reg_panalty)

                        ## Define
                        if ml_model == 'xgb':
                            model = XGBClassifier(**best_para, n_jobs = core_cnt, random_state = seed)
                            model.fit(
                                train_df[x_var],
                                train_df[y_var],
                                early_stopping_rounds = 15,
                                eval_set = [(train_df[x_var], train_df[y_var]),
                                            (valid_df[x_var], valid_df[y_var])],
                                verbose = False
                            )
                        else:
                            model = LGBMClassifier(**best_para, n_jobs = core_cnt, random_state = seed)
                            model.fit(
                                train_df[x_var],
                                train_df[y_var],
                                eval_set = [(train_df[x_var], train_df[y_var]),
                                            (valid_df[x_var], valid_df[y_var])],
                                callbacks = [
                                early_stopping(
                                    stopping_rounds = 15,
                                    verbose = False
                                    )
                                ]
                            )

                        ## Number of Estimator
                        if ml_model == 'xgb':
                            nestimator_ls.append(model.best_iteration)
                        else:
                            nestimator_ls.append(model.best_iteration_)

                        ## Score
                        score = log_loss(
                            valid_df[y_var], 
                            model.predict_proba(valid_df[x_var])
                        )
                        score_ls.append(score)

                ## L1&L2 regularization's Result
                reg_result = pd.DataFrame({
                    'reg_panalty': reg_ls, 'n_estimator': nestimator_ls, 'score': score_ls
                    }
                )

                ## Filtering by n_estimator
                if len(reg_result[reg_result['n_estimator'] >= 100] > 0):
                    reg_result = reg_result[reg_result['n_estimator'] >= 100]
                elif len(reg_result[reg_result['n_estimator'] >= 50] > 0):
                    reg_result = reg_result[reg_result['n_estimator'] >= 50]
                else:
                    reg_result = pd.DataFrame()

                if len(reg_result) > 0:
                    ## Sort & Initialize index
                    reg_result_agg = (
                        reg_result\
                            .groupby('reg_panalty')\
                            .mean()\
                            .reset_index()\
                            .sort_values(['score', 'reg_panalty'], ascending = [True, True])\
                            .reset_index()
                    )

                    ## Final Learning rate & n_estimator
                    best_reg = reg_result_agg.iloc[0]['reg_panalty']

                    ## Add final parameter
                    best_para[reg_type] = max(best_reg, 0)
                else:
                    ## Default setting
                    if reg_type == 'reg_alpha':
                        best_para[reg_type] = 0
                    else:
                        if ml_model == 'xgb':
                            best_para[reg_type] = 1
                        else:
                            best_para[reg_type] = 0


        ## Number of Estimator
        if ml_model in ['xgb', 'lgbm']:
            
            print('Optimizing Number of Estimators & Learning Rate')
            
            ## Initial n_estimators
            best_para['n_estimators'] = 1000

            ## Learning rate List
            learn_ls      = []
            ## Number of Estimator List
            nestimator_ls = []
            ## Score List
            score_ls      = []

            ## Train / Test Data
            kfold_data = KFold(n_splits = cv_num, shuffle = True, random_state = seed + 2)

            for i0, (train_idx, valid_index) in enumerate(kfold_data.split(total_data[x_var])):
                ## Split Train/Test
                train_df = total_data.iloc[train_idx]
                valid_df = total_data.iloc[valid_index]
                
                for lean_rate in lean_rate_ls:
                    ## Add learning rate
                    best_para['learning_rate'] = lean_rate
                    learn_ls.append(lean_rate)

                    ## Define
                    if ml_model == 'xgb':
                        model = XGBClassifier(**best_para, n_jobs = core_cnt, random_state = seed)
                        ## Fit
                        model.fit(
                            train_df[x_var],
                            train_df[y_var],
                            early_stopping_rounds = 50,
                            eval_set = [(train_df[x_var], train_df[y_var]),
                                        (valid_df[x_var], valid_df[y_var])],
                            verbose = False
                        )
                    else:
                        model = LGBMClassifier(**best_para, n_jobs = core_cnt, random_state = seed)
                        model.fit(
                                train_df[x_var],
                                train_df[y_var],
                                eval_set = [(train_df[x_var], train_df[y_var]),
                                            (valid_df[x_var], valid_df[y_var])],
                                callbacks = [
                                early_stopping(
                                    stopping_rounds = 50,
                                    verbose = False
                                    )
                                ]
                            )

                    

                    ## Number of Estimator
                    if ml_model == 'xgb':
                        nestimator_ls.append(model.best_iteration)
                    else:
                        nestimator_ls.append(model.best_iteration_)

                    ## Score
                    if ml_model == 'xgb':
                        score = log_loss(
                            valid_df[y_var], 
                            model.predict_proba(valid_df[x_var])
                        )
                    else:
                        score = log_loss(
                            valid_df[y_var], 
                            model.predict_proba(valid_df[x_var])
                        ) 
                    score_ls.append(score)

            ## Learning Rate & N_Estimator's Result
            lr_nestimator_result = pd.DataFrame({
                'lr': learn_ls, 'n_estimator': nestimator_ls, 'score': score_ls
                }
            )

            ## Filtering with n_estimator
            if len(lr_nestimator_result[lr_nestimator_result['n_estimator'] >= 100]) > 0:
                ## n_estimator >= 100
                lr_nestimator_result = lr_nestimator_result[lr_nestimator_result['n_estimator'] >= 100]
            elif len(lr_nestimator_result[lr_nestimator_result['n_estimator'] >= 50]) > 0:
                ## n_estimator >= 50
                lr_nestimator_result = lr_nestimator_result[lr_nestimator_result['n_estimator'] >= 50]

            ## Sort & Initialize index
            lr_nestimator_result_agg = lr_nestimator_result.groupby('lr').mean().reset_index().sort_values(['score', 'lr'], ascending = [True, False]).reset_index()

            ## Final Learning rate & n_estimator
            best_lr    = lr_nestimator_result_agg.iloc[0]['lr']
            best_nesti = int(round(lr_nestimator_result_agg.iloc[0]['n_estimator'], -1))

            ## Add final parameter
            best_para['learning_rate'] = best_lr
            best_para['n_estimators']  = int(max(best_nesti, 30))

        elif ml_model == 'rf':
            
            print('Optimizing Number of Estimators with OOB Score')
            
            ## Result List
            n_estimators_ls = [100]
            ## OOB Score(Mean squared error)
            score_ls        = []

            ## Initial n_estimators
            best_para['n_estimators'] = int(n_estimators_ls[0])
            ## Define
            model = RandomForestClassifier(
                **best_para, 
                n_jobs = core_cnt, 
                random_state = seed, 
                oob_score = True, 
                warm_start = True
            )
            ## Fit
            model.fit(total_data[x_var], total_data[y_var])
            ## Score
            score = model.oob_score_
            score_ls.append(score)

            stop_cnt = 0
            while True:
                try_n_estimator = n_estimators_ls[-1] + 10

                if try_n_estimator > 1000:
                    break
                n_estimators_ls.append(try_n_estimator)

                ## Add n_estimators
                model.n_estimators += 10
                ## Fit
                model.fit(total_data[x_var], total_data[y_var])
                ## Score
                score = model.oob_score_
                score_ls.append(score)

                ## Early stopping
                if score == min(score_ls):
                    stop_cnt = 0
                else:
                    stop_cnt += 1

                if stop_cnt > 10:
                    break
            ## Result
            best_para['n_estimators'] = int(n_estimators_ls[score_ls.index(min(score_ls))])

        ## Print best hyper-parameter
        if verbose:

            para_str = ""
            for i0, (k0, v0) in enumerate(best_para.items()):
                if i0 != (len(best_para) - 1):
                    para_str += str(k0) + ": " + str(v0) + ', '
                else:
                    para_str += str(k0) + ": " + str(v0)

            print("Best Para({}): {}".format(ml_model.upper(), para_str))

    except Exception as e:
        print('Error in fnOpt_HyperPara')
        print(e)
        
        raise Exception('Check error')

    if verbose:
        end_time = datetime.datetime.now()
        print('Optimizing Hyper-Para({}) Elapsed Time: {} Minutes'.format(
            ml_model.upper(), round((end_time - st_time).seconds / 60, 2)
            )
        )

    return trials, best_para