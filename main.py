
import numpy as np
import pandas as pd
import datetime

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from model.feature_select import fnFeatSelect_RandVar
from model.optimize_hyperparameter import fnOpt_HyperPara
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from hyperopt import fmin, hp, STATUS_OK, Trials, tpe

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


####################################################################################
#### Data Load

## Sample Data
from sklearn.datasets import load_breast_cancer

## Loader
cancer = load_breast_cancer()
## Feature Data
dataX = pd.DataFrame(
    cancer.data,
    columns = cancer.feature_names
)
featureList = dataX.columns.tolist()

## Tareget Data
TargetNM = 'CANCER'
dataY = pd.DataFrame(
    cancer.target,
    columns = [TargetNM]
)

dataY[f'{TargetNM}_LABEL'] = dataY[TargetNM].apply(
    lambda x: cancer.target_names[0] if x == 0 else cancer.target_names[1]
)

## Union: Total Data
totalData = pd.concat([dataX, dataY[TargetNM]], axis = 1)

print('Data: {}'.format(totalData.shape))
print('Feature Types')
print(totalData.dtypes, '\n')

print('Malignant(악성): {}'.format(dataY[dataY[TargetNM] == 0].shape[0]))
print('Benign(양성): {}'.format(dataY[dataY[TargetNM] == 1].shape[0]))


####################################################################################
#### Train / Test 분할(8:2)

(
    trainX, 
    testX, 
    trainY, 
    testY
) = train_test_split(
    totalData[featureList], 
    totalData[TargetNM], 
    test_size = 0.2, 
    random_state = 1000,
    stratify = totalData[TargetNM],
    shuffle = True
)

print('Train Data: {}'.format(trainX.shape))
print('Test Data: {}'.format(testX.shape))


####################################################################################
#### 전처리

## 변수 Scale
scaler = MinMaxScaler()
trainScale = scaler.fit_transform(trainX[featureList])
testScale = scaler.transform(testX[featureList])

## DataFrame 변환
trainScale_DF = pd.DataFrame(
    trainScale,
    columns = featureList
)
trainScale_DF[TargetNM] = trainY.values
testScale_DF = pd.DataFrame(
    testScale,
    columns = featureList
)
testScale_DF[TargetNM] = testY.values

print('Scaled Train Info')
print(trainScale_DF.describe(), '\n')
print('Scaled Test Info')
print(testScale_DF.describe(), '\n')


####################################################################################
#### 변수선택

## 임의로 생성한 Random변수들보다 변수중요도가 떨어지는 변수들을 제거하는 로직
feat_RandSelct_LS = fnFeatSelect_RandVar(
    df_x = trainScale_DF[featureList], 
    df_y = trainScale_DF[TargetNM], 
    x_var = featureList, 
    core_cnt = -1
)
deleteFeatureLS = list(set(featureList) - set(feat_RandSelct_LS))
print('Feauter List with Rand-Select: {}'.format(feat_RandSelct_LS))

if len(deleteFeatureLS) > 0:
    print('Delete Feature List: {}'.format(deleteFeatureLS))

## 최종 변수 리스트
feat_Final_LS = feat_RandSelct_LS


####################################################################################
#### Hyper-parameter 최적화

MODEL_NM = 'xgb'
H_PARA_SPACE = {
    'max_depth': hp.uniform("max_depth", 1, 30),
    'min_child_weight': hp.loguniform('min_child_weight', -3, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.loguniform('gamma', -10, 10)
}

TrialResult, BestPara = fnOpt_HyperPara(
    total_data = trainScale_DF, 
    x_var = feat_Final_LS, 
    y_var = TargetNM, 
    space = H_PARA_SPACE, 
    lean_rate_ls = [0.001, 0.01, 0.1], 
    ml_model = MODEL_NM, 
    core_cnt = -1, 
    cv_num = 3, 
    max_evals = 30, 
    seed = 1000, 
    verbose = True,
)


####################################################################################
#### 최종 모델학습 및 예측
finalModel = XGBClassifier(**BestPara)
finalModel.fit(
    trainScale_DF[feat_Final_LS], 
    trainScale_DF[TargetNM],
    early_stopping_rounds = 50,
    eval_set = [(trainScale_DF[feat_Final_LS], trainScale_DF[TargetNM])],
    verbose = False
)

## Predict
finalPredict = finalModel.predict(testScale_DF[feat_Final_LS])

## Score
SCORE_ACC = accuracy_score(
    testScale_DF[TargetNM],
    finalPredict
)
SCORE_PRECISION = precision_score(
    testScale_DF[TargetNM],
    finalPredict
)
SCORE_RECALL = recall_score(
    testScale_DF[TargetNM],
    finalPredict
)
SCORE_F1 = f1_score(
    testScale_DF[TargetNM],
    finalPredict
)
confusionMatrix = confusion_matrix(
    testScale_DF[TargetNM],
    finalPredict
)

print(f'Accuracy: {SCORE_ACC}')
print(f'Precission: {SCORE_PRECISION}')
print(f'Recall: {SCORE_RECALL}')
print(f'F1 Score: {SCORE_F1}')
print(confusionMatrix)

sns.heatmap(confusionMatrix, annot=True, cmap='Blues')
plt.show()