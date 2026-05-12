import sys

import pandas as pd



sys.path.append("../")


import src.features

from pathlib import Path



from src.features import generate_features as gen
from src.features import get_dataset, get_periods

from src.evaluate_prediction import prediction_df, evaluate_prediction, evaluate_Models


import src.Model_circus

def run_model(Models,Simulation_start_date,Simulation_end_date, train_years = 20,
                valid_years = 1,
                step_years = 1,max_steps = 50  ):



    Periods= get_periods(Simulation_start_date,Simulation_end_date, train_years = train_years, valid_years = valid_years, step_years = step_years)


    rows=[]
    k=0
    for i in Periods.index:
        Dataset= get_dataset(Periods.iloc[i])
        Z_train=Dataset["Z_train"]
        y_train=Dataset["y_train"]
        Z_valid=Dataset["Z_valid"]
        y_valid=Dataset["y_valid"]



        year=Periods.iloc[i]['valid_start'].year


        for name, model in Models.items():
            #print(Z_train,y_train)
            model.fit(Z_train,y_train)
            #y_train_predict = model.predict(Z_train)
            #evaluate_prediction(y_train_predict, y_train)
            y_valid_predict = model.predict(Z_valid)
            mqe_value,r2_value=evaluate_prediction(y_valid_predict, y_valid)
            rows.append({
                        "year": year,
                        "model": name,
                        "R2_valid": round(r2_value,4),
                        "mqe_valid": round(mqe_value,4),
                    })

        k += 1
        if k>=max_steps:
            break
        #print(year)
    eval_df = pd.DataFrame(rows)
    eval_summary= eval_df.pivot(index="year", columns="model", values="R2_valid")



    eval_mean =eval_summary.mean(axis=0).to_frame().T
    eval_mean.index = ["average"]



    return eval_summary, eval_mean








if __name__ == "__main__":
    Models = src.Model_circus.build_models()
    eval_summary_valid, eval_mean_valid=run_model(Models,"1970-01-31","1999-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps = 10)
    print('Valid run finished')
    eval_summary_test, eval_mean_test=run_model(Models,"1980-01-31","2020-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps = 10)
    print('test run finished')


    dir_eval = "../data/results/evaluation"
    path_eval = Path(dir_eval)
    path_eval.mkdir(parents=True, exist_ok=True)

    eval_summary_valid.to_csv(path_eval/'valid_eval_summary.csv')
    eval_mean_valid.to_csv(path_eval/'valid_eval_mean.csv')
    eval_summary_test.to_csv(path_eval/'test_eval_summary.csv')
    eval_mean_test.to_csv(path_eval/'test_eval_mean.csv')











