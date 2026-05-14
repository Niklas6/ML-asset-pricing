import sys

import pandas as pd

sys.path.append("../")
import src.features
from pathlib import Path

from src.features import generate_features as gen
from src.features import get_dataset, get_periods

from src.evaluate_prediction import  evaluate_prediction


import src.Model_circus

def run_model(Models,Simulation_start_date,Simulation_end_date, train_years = 20,
                valid_years = 1,
                step_years = 1,max_steps = 50 ,print_years=False ):



    Periods= get_periods(Simulation_start_date,Simulation_end_date, train_years = train_years, valid_years = valid_years, step_years = step_years)


    rows=[]
    k=0
    for i in Periods.index:
        Dataset= get_dataset(Periods.iloc[i])
        Z_train=Dataset["Z_train"]
        y_train=Dataset["y_train"]

        Z_valid=Dataset["Z_valid"]
        y_valid=Dataset["y_valid"]

        risk_free_rate = pd.read_csv('../data/raw/risk_free_rate.csv', index_col='Date', parse_dates=True)

        year=Periods.iloc[i]['valid_start'].year


        '''y_valid_rf = y_valid.copy()

        for ind in y_valid.index:
            y_valid_rf[ind] = y_valid[ind] + risk_free_rate.loc[ind[0]]'''






        for name, model in Models.items():
            model.fit(Z_train,y_train)
            y_valid_predict = model.predict(Z_valid)
            mqe_value,r2_value=evaluate_prediction(y_valid_predict,y_valid,y_valid_riskfree)
            rows.append({
                        "year": year,
                        "model": name,
                        "R2_valid": r2_value,
                        "mqe_valid": mqe_value,
                    })

        k += 1
        if print_years:
            print(year)
        if k>=max_steps:
            break
        #print(year)
    eval_df = pd.DataFrame(rows)
    eval_summary= eval_df.pivot(index="year", columns="model", values="R2_valid")

    eval_stat = eval_summary.agg(['mean', 'std', 'min'])


    eval_summary=eval_summary.round(3)
    eval_stat=eval_stat.round(3)

    return eval_summary, eval_stat








if __name__ == "__main__":
    Models = src.Model_circus.build_models()
    eval_summary_valid, eval_mean_valid=run_model(Models,"1970-01-31","1999-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps = 10,print_years=True)
    print('Valid run finished')

    dir_eval = "../data/results/evaluation"
    path_eval = Path(dir_eval)
    path_eval.mkdir(parents=True, exist_ok=True)

    eval_summary_valid.to_csv(path_eval / 'valid_eval_summary.csv')
    eval_mean_valid.to_csv(path_eval / 'valid_eval_mean.csv')

    eval_summary_test1, eval_mean_test1=run_model(Models,"1980-01-31","2009-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps = 10,print_years=True)
    print('test 1 run finished')

    eval_summary_test1.to_csv(path_eval/'test1_eval_summary.csv')
    eval_mean_test1.to_csv(path_eval/'test1_eval_mean.csv')

    eval_summary_test2, eval_mean_test2=run_model(Models,"1990-01-31","2019-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps = 10,print_years=True)
    print('test 2 run finished')

    eval_summary_test2.to_csv(path_eval/'test2_eval_summary.csv')
    eval_mean_test2.to_csv(path_eval/'test2_eval_mean.csv')














