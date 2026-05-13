
import pandas as pd
import numpy as np
import importlib
import time


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor as XGB
from sklearn.ensemble import RandomForestRegressor as RF

from sklearn.tree import DecisionTreeRegressor as DTR

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression as LR

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


from sklearn.linear_model import MultiTaskElasticNet
from sklearn.ensemble import ExtraTreesRegressor


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


import statsmodels.api as sm







def build_models():
    OLSModel = sm.OLS
    RidgeModel = Ridge(alpha=10.0)

    ElasticNetModel = MultiTaskElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000)

    ExtraTreesModel = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


    DTRmodel=DTR(max_leaf_nodes=20)
    RFmodel=RF(n_estimators=100)


    XGBmodel1=XGB(
    n_estimators=100,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    reg_alpha=1,
    random_state=42
    )

    XGBmodel2=XGB(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    reg_alpha=1,
    random_state=42
    )

    NN1 = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(16,),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        )
    )
    NN2 = MLPRegressor(
        hidden_layer_sizes=(64),
        activation="relu",
    max_iter=400,
    random_state=42
    )
    NN3 = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        random_state=42
    )
    NN4 = MLPRegressor(
        hidden_layer_sizes=(64, 32,16),
        activation="relu",
        max_iter=200,
        random_state=42
    )



    Models= {
        #'OLS model': OLSModel,
        'Ridge model': RidgeModel,
        #'Elastic net model': ElasticNetModel,
        'Extra tree model': ExtraTreesModel,
        'DTR model': DTRmodel,
        'RF model': RFmodel,
        'XGBmodel1': XGBmodel1,
        'XGBmodel2': XGBmodel2,
        'NN1': NN1,
        #'NN2': NN2,
        #'NN3': NN3,
        #'NN4': NN4,

}

    return Models