

import pandas as pd

import numpy as np




from sklearn.metrics import mean_squared_error





def prediction_df(y_pred, y):
    return pd.DataFrame(y_pred,columns=y.columns, index=y.index)






def evaluate_prediction(y_pred, y,yrfr,s: str='data'):
    mqe_pred=np.sqrt(mean_squared_error(y_pred,y))
    #zero_rmse = (y**2).sum().sum() #np.sqrt((y ** 2).mean().mean())

    return(mqe_pred,1-((y_pred-y)**2).sum().sum()/((y-yrfr)**2).sum().sum())
    #print('Prediction on '+s+' has the mean sqare error',mqe_pred, ' compared to guess model ',zero_rmse)
    #print('In the R^2 metric: ', 1-mqe_pred/zero_rmse)



def evaluate_Models(Z_train,y_train,Z_valid,y_valid,Model):
    rows= []

    rows.append({    'name': 'zero_guess',
        'mqe_train': np.sqrt((y_train ** 2).mean().mean()),
        'R2_train': 0,
        'mqe_valid': np.sqrt((y_valid ** 2).mean().mean()),
        'R2_valid': 0
         })


    for name, model in Model.items():
    #print(model.__class__.__name__)
        y_train_predict=prediction_df(model.predict(Z_train),y_train)
        y_valid_predict=prediction_df(model.predict(Z_valid),y_valid)


        mqe_train,R2_train=evaluate_prediction(y_train_predict, y_train, 'train data')
        mqe_valid,R2_valid=evaluate_prediction(y_valid_predict, y_valid, 'valid data')
        rows.append({    'name': name,
            'mqe_train': mqe_train ,
            'R2_train': R2_train,
            'mqe_valid': mqe_valid,
            'R2_valid': R2_valid
                })
    return pd.DataFrame(rows).set_index("name")