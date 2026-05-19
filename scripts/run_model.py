import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from pathlib import Path

from src.features import get_dataset, get_periods

from src.evaluate_prediction import  evaluate_prediction

from src.portfolio import Portfolio, uniform_25, revenue_pred,portfolio_weight, backtest_diag


import src.Models

from dataclasses import dataclass, field

@dataclass(slots=True)
class ModelData:
    name: str
    revenue_df: pd.DataFrame


def run_model(Models,Simulation_start_date,Simulation_end_date, train_years = 20,
                valid_years = 1,
                step_years = 1,max_steps = 50 ,print_years=False ):

    #print(Models.keys())

    Periods= get_periods(Simulation_start_date,Simulation_end_date, train_years = train_years, valid_years = valid_years, step_years = step_years)

    rows=[]
    k=0

    #print(ModelDict)
    Yearly_revenue=pd.DataFrame(index=pd.Index([], name="year"))

    Yearly_volat=pd.DataFrame()
    for i in Periods.index:
        Dataset= get_dataset(Periods.iloc[i])
        Z_train=Dataset["Z_train"]
        y_train=Dataset["y_train"]

        Z_valid=Dataset["Z_valid"]
        y_valid=Dataset["y_valid"]


        risk_free_rate = pd.read_csv(PROJECT_ROOT /'data'/'raw'/'risk_free_rate.csv', index_col='Date', parse_dates=True)

        year=Periods.iloc[i]['valid_start'].year


        '''y_valid_rf = y_valid.copy()

        for ind in y_valid.index:
            y_valid_rf[ind] = y_valid[ind] + risk_free_rate.loc[ind[0]]'''

        y_valid_riskfree = y_valid.copy()
        for ind in y_valid_riskfree.index:
            y_valid_riskfree[ind] = risk_free_rate.shift(-1).loc[ind[0]].iloc[0]




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
        montly_revenue_df = pd.DataFrame(index=Z_valid.index.get_level_values(0).unique(), columns=Models.keys(), dtype=float)
        for month in Z_valid.index.get_level_values(0).unique():
            for name, model in Models.items():
                y_valid_month = y_valid.loc[month, :]
                y_pred_month = model.predict(Z_valid.loc[month, :])
                y_pred_month = pd.DataFrame({'prediction': y_pred_month}, index=Z_valid.loc[month, :].index)
                y_pred_month = y_pred_month.sort_values(by=['prediction'])
                port = portfolio_weight(y_pred_month)
                revenue= revenue_pred(port, y_valid_month)
                #print(month,name)
                montly_revenue_df.loc[month,name]=revenue


        ydf = ((1 + montly_revenue_df).prod(axis=0) - 1).to_frame().T
        ydf.index = pd.Index([year], name="year")
        Yearly_revenue = pd.concat([Yearly_revenue, ydf], axis=0)
        #Yearly_revenue.index ="year"



        k += 1
        if print_years:
            print('year: ',year,', ', '# companies:',y_valid.index.get_level_values(1).nunique())
        if k>=max_steps:
            break
        #print(year)
    eval_df = pd.DataFrame(rows)
    eval_summary= eval_df.pivot(index="year", columns="model", values="R2_valid")

    eval_stat = eval_summary.agg(['mean', 'std', 'min'])


    eval_summary=eval_summary.round(3)
    eval_stat=eval_stat.round(3)
    #print(Yearly_revenue)
    #Yearly_revenue.set_index('year', inplace=True)
    return {'eval_summary':eval_summary,
            'eval_stat':eval_stat,
            'yearly_revenue':Yearly_revenue,
            'backtest_data': backtest_diag(Yearly_revenue)
            }








if __name__ == "__main__":
    Models = src.Models.build_models()
    valid_run=run_model(Models,"1970-01-31","1999-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps = 10,print_years=True)
    print('Valid run finished')

    eval_summary_valid=valid_run['eval_summary']
    eval_mean_valid=valid_run['eval_stat']
    Yearly_revenue_valid=valid_run['yearly_revenue']
    backtest_data_valid=valid_run['backtest_data']


    test1_run=run_model(Models,"1980-01-31","2009-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps =10,print_years=True)
    print('First test run finished')
    eval_summary_test1=test1_run['eval_summary']
    eval_mean_test1=test1_run['eval_stat']
    Yearly_revenue_test1=test1_run['yearly_revenue']
    backtest_data_test1=test1_run['backtest_data']


    test2_run=run_model(Models,"1990-01-31","2019-12-31",train_years = 20, valid_years = 1,step_years = 1 ,max_steps = 10,print_years=True)
    print('Second test run finished')
    eval_summary_test2=test2_run['eval_summary']
    eval_mean_test2=test2_run['eval_stat']
    Yearly_revenue_test2=test2_run['yearly_revenue']
    backtest_data_test2=test2_run['backtest_data']

    dir_eval = PROJECT_ROOT /'data'/'results'
    path_eval = Path(dir_eval)
    path_eval.mkdir(parents=True, exist_ok=True)

    prediction_decade = pd.DataFrame({
        '1990-1999': eval_mean_valid.iloc[0],
        '2000-2009': eval_mean_test1.iloc[0],
        '2010-2019': eval_mean_test2.iloc[0],
    }).T
    prediction_decade=prediction_decade.loc[:,Models.keys()]
    prediction_decade.to_csv(path_eval / 'prediction_decade.csv')

    backtest_decade= pd.DataFrame({
        "1990-1999": ((1 + Yearly_revenue_valid).prod(axis=0) ** (1 / 10) - 1).round(3),
        "2000-2009": ((1 + Yearly_revenue_test1).prod(axis=0) ** (1 / 10) - 1).round(3),
        "2010-2019": ((1 + Yearly_revenue_test2).prod(axis=0) ** (1 / 10) - 1).round(3),
    }).T
    ''''1990-1999': ((1 + Yearly_revenue_valid).prod(axis=0) **(1/10)- 1).to_frame().T,# (Yearly_revenue_valid.sum()/10).round(3),
            '2000-2009': ((1 + Yearly_revenue_test1).prod(axis=0) **(1/10)- 1).to_frame().T, #(Yearly_revenue_test1.sum()/10).round(3),
            '2010-2019': ((1 + Yearly_revenue_test2).prod(axis=0) **(1/10)- 1).to_frame().T# (Yearly_revenue_test2.sum()/10).round(3),'''
    backtest_decade=backtest_decade.loc[:,Models.keys()]
    backtest_decade.to_csv(path_eval / 'backtest_decade.csv')


    backtest_yearly = pd.concat([Yearly_revenue_valid,Yearly_revenue_test1,Yearly_revenue_test2], axis=0).rename_axis("year").round(3)
    backtest_yearly=backtest_yearly.loc[:,Models.keys()]
    backtest_yearly.to_csv(path_eval / 'backtest_yearly.csv')


    prediction_yearly= pd.concat([eval_summary_valid, eval_summary_test1, eval_summary_test2], axis=0).round(3)
    prediction_yearly = prediction_yearly.loc[:, Models.keys()]
    prediction_yearly.to_csv(path_eval / 'prediction_yearly.csv')


    backtest_data = pd.concat([backtest_data_valid,backtest_data_test1,backtest_data_test2], axis=0).rename_axis("year").round(3)
    backtest_data.to_csv(path_eval / 'backtest_data.csv')

















