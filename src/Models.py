
from xgboost import XGBRegressor as XGB
from sklearn.ensemble import RandomForestRegressor as RF

from sklearn.tree import DecisionTreeRegressor as DTR

from sklearn.linear_model import Ridge


from sklearn.linear_model import MultiTaskElasticNet
from sklearn.ensemble import ExtraTreesRegressor


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor






from lightgbm import LGBMRegressor as LGBM





def build_models():
    RidgeModel = Ridge(
                alpha=1,
                fit_intercept=False,
                solver="auto",
                max_iter=10,
                tol=1e-4,
                random_state=42
        )



    ExtraTreesModel = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


    RFmodel=RF(
        n_estimators=500,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )



    XGBmodel=XGB(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    reg_alpha=1,
    random_state=42
    )
    LGBMmodel = LGBM(
        n_estimators=200 ,
        learning_rate=0.02,
        max_depth=2,
        subsample=0.6,
        colsample_bytree=0.8,
        reg_lambda=10,
        reg_alpha=1,
        random_state=42,
        verbose=-1
    )

    # Models which could be added later
    ElasticNetModel = MultiTaskElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000)
    DTRmodel=DTR(max_leaf_nodes=20)

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


    Models= {
        'Ridge model': RidgeModel,
        'RF model': RFmodel,
        'Extra tree model': ExtraTreesModel,
        'XGB model': XGBmodel,
        'LGBM model': LGBMmodel

}

    return Models