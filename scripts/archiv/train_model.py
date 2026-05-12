import sys

import pandas as pd



sys.path.append("../../")


import src.features

from pathlib import Path



from src.features import generate_features as gen
from src.features import get_dataset, get_periods

from src.evaluate_prediction import prediction_df, evaluate_prediction, evaluate_Models


import src.Model_circus

def run_model(train_years = 20,
                valid_years = 1,
                test_years = 1,
                step_years = 1,max_steps = 50  ) -> None:



    Periods= get_periods("1970-01-31","2020-12-31", train_years = train_years, valid_years = valid_years, test_years = test_years, step_years = step_years)
    Models = src.Model_circus.build_models()


    k=max_steps
    dir_yearly= "../data/results/evaluation/yearly"
    path_yearly= Path(dir_yearly)
    path_yearly.mkdir(parents=True, exist_ok=True)
    for file in path_yearly.glob("*.csv"):
        file.unlink()

    rows=[]
    k=0


    #print(Periods)

    for i in Periods.index:
        Dataset= get_dataset(Periods.iloc[i])
        Z_train=Dataset["Z_train"]
        y_train=Dataset["y_train"]
        Z_valid=Dataset["Z_valid"]
        y_valid=Dataset["y_valid"]
        for names, model in Models.items():
            model.fit(Z_train,y_train)

        eval= evaluate_Models(Z_train,y_train,Z_valid,y_valid,Models)
        year=Periods.iloc[i]['valid_start'].year
        print(year)
        s= 'eval'+str(year)+".csv"

        sorted_eval = eval.sort_values(by=['R2_valid'], ascending=False)
        sorted_eval.to_csv(path_yearly/s)


        #print(Models.keys())
        if k==0:
            eval_summary=eval.loc[:,['R2_valid']].rename(columns={ "R2_valid":( str(year)+"_R2")})
        else:
            eval_summary=pd.concat([eval_summary,eval.loc[:,['R2_valid']].rename(columns={ "R2_valid": (str(year)+"_R2")})],axis=1)
        if k>=max_steps-1:
            break
        k+=1
    dir_eval = "../../data/results/evaluation"
    path_eval = Path(dir_eval)
    path_eval.mkdir(parents=True, exist_ok=True)
    print(eval_summary.index)

    eval_summary.to_csv(path_eval/'eval_summary.csv')
    eval_mean=eval_summary.mean(axis=1).rename("mean_R2")
    #eval_mean=eval_mean.sort_values(by=['R2_valid'], ascending=False)
    eval_mean.to_csv(path_eval/'eval_mean.csv', header=True)
    print('finished')

if __name__ == "__main__":
    run_model(train_years = 20,
                valid_years = 1,
                test_years = 1,
                step_years = 1 ,max_steps = 2)