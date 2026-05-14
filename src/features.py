
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



import pandas as pd
import numpy as np


def generate_features(load_start_time="1965-01-31", load_end_time="2025-12-31"):
    df = pd.read_csv(PROJECT_ROOT/'data'/'raw'/'prices.csv', index_col='Date', parse_dates=True)
    #ts = pd.Timestamp(train_start_date)
    #data_load_time = ts - pd.DateOffset(years=5)
    #data_load_time=data_load_time.strftime("%Y-%m-%d")
    #load_start_time="1965-01-31"
    #load_end_time ="2025-12-31"
    df = df.loc[load_start_time:load_end_time]


    df_bench = pd.read_csv(PROJECT_ROOT/'data'/'raw'/'prices_bench.csv', index_col='Date', parse_dates=True)
    df_bench = df_bench.loc[load_start_time:load_end_time]

    risk_free_rate = pd.read_csv(PROJECT_ROOT/'data'/'raw'/'risk_free_rate.csv', index_col='Date', parse_dates=True)


    price_monthly = df.resample("ME").last()
    #valid_names =(price_monthly.iloc[0]!=0) & (price_monthly.iloc[0].notna()) & (price_monthly.iloc[0].notnull())
    #price_monthly=price_monthly.loc[:,valid_names]

    ret_1m=price_monthly.pct_change(1)
    ret_3m=price_monthly.pct_change(3)
    ret_6m=price_monthly.pct_change(6)
    ret_12m=price_monthly.pct_change(12)

    daily_ret = df.pct_change()
    std_1m = daily_ret.resample("ME").std()

    std_3m = daily_ret.rolling(window=63).std().resample("ME").last()
    std_6m = daily_ret.rolling(window=126).std().resample("ME").last()
    std_12m = daily_ret.rolling(window=252).std().resample("ME").last()
    ret_next_month = ret_1m.shift(-1)


    bench_price = df_bench.iloc[:, 0]
    price_monthly_bench = bench_price.resample("ME").last()

    ret_1m_bench=price_monthly_bench.pct_change(1)
    ret_1y_bench=price_monthly_bench.pct_change(12)

    daily_ret_bench  = bench_price.pct_change()
    std_1m_bench  = daily_ret_bench.resample("ME").std()
    std_1y_bench = daily_ret_bench.rolling(window=252).std().resample("ME").last()

    rolling_window = 252
    min_periods = 126

    beta_1y_daily = (
        daily_ret
        .rolling(window=rolling_window, min_periods=min_periods)
        .cov(daily_ret_bench)
        .div(
            daily_ret_bench.rolling(window=rolling_window, min_periods=min_periods).var(),
            axis=0
        )
    )

    beta_1y = beta_1y_daily.resample("ME").last()

    beta_ret_1y=pd.DataFrame([beta_1y.loc[:,col]*ret_1m_bench for col in beta_1y.columns]).T
    beta_ret_1y.columns= beta_1y.columns


    beta_std_1y=pd.DataFrame([beta_1y.loc[:,col]*std_1m_bench for col in beta_1y.columns]).T
    beta_std_1y.columns= beta_1y.columns

    adj_ret_1m = pd.DataFrame(
        [ret_1m.loc[:, col] - risk_free_rate.loc[ret_1m.index, 'rf_monthly'] for col in
         ret_1m.columns]).T
    adj_ret_1m.columns = ret_1m.columns


    adj_ret_next_month=pd.DataFrame([ret_next_month.loc[:, col]-risk_free_rate.loc[ret_next_month.index,'rf_monthly']  for col in ret_next_month.columns]).T
    adj_ret_next_month.columns= ret_next_month.columns

    #print(adj_ret_next_month)
    #print(ret_next_month)

    Stock_propeties = pd.concat(
        {
            "price_m": price_monthly,
            #'exists': exists,
            "ret_m1": ret_1m,
            #"adj_ret_1m": adj_ret_1m,
            "ret_m3": ret_3m,
            "ret_m6": ret_6m,
            "ret_y1": ret_12m,
            'std_m1': std_1m,
            'std_m3': std_3m,
            'std_m6': std_6m,
            'std_y1': std_12m,
            'beta_ret_1y': beta_ret_1y,
            'beta_std_1y': beta_std_1y,
            'ret_next_month': ret_next_month,
            #'adj_ret_next_month': adj_ret_next_month,#-risk_free_rate,
        },
        axis=1
    )

    Stock_propeties = Stock_propeties.swaplevel(axis=1).sort_index(axis=1)

    Stock_propeties.index.name = "date"
    Stock_propeties.columns.names = ["stock", "feature"]

    Stock_propeties = (
        Stock_propeties
        .stack(level="stock")
        .reset_index()
    )
    #Stock_propeties["ret_1m_bench"] = Stock_propeties["date"].map(ret_1m_bench)
    #Stock_propeties["beta*ret_1y_bench"] = Stock_propeties["date"].map(ret_1y_bench)
    #Stock_propeties["beta*std_1y_bench"] = Stock_propeties["date"].map(std_1y_bench)
    #averages = averages.sort_values(["date", "stock"])

    #feature_cols = ["price_m","ret_m1", "ret_m3", "ret_m6", "ret_y1", 'std_m1', 'std_m3', 'std_m6', 'std_y1','beta_1y',"ret_1m_bench","ret_1y_bench",'ret_next_month']

    Stock_propeties = Stock_propeties.replace([np.inf, -np.inf], np.nan)
    Stock_propeties = Stock_propeties.dropna()
    Stock_propeties = Stock_propeties.set_index(['date',	'stock'])
    '''feature_order = [
        "price_m",
        "ret_m1",
        "ret_m3",
        "ret_m6",
        "ret_y1",
        "std_m1",
        "std_m3",
        "std_m6",
        "std_y1",
        'beta_ret_1y'
        'beta_std_1y',
        "ret_next_month",
    ]
    Stock_propeties = Stock_propeties[feature_order]'''

    return Stock_propeties




def get_dataset(period):
    train_start_date=period['train_start'].strftime("%Y-%m-%d")
    train_end_date=period['train_end'].strftime("%Y-%m-%d")
    valid_start_date=period['valid_start'].strftime("%Y-%m-%d")
    valid_end_date=period['valid_end'].strftime("%Y-%m-%d")

    X = pd.read_csv(PROJECT_ROOT/'data'/'processed'/'X.csv').set_index(['date','stock'])
    #print( generate_features('1965-01-31',"2025-12-31"))
    #print(pd.read_csv(PROJECT_ROOT/'data'/'processed'/'X.csv').set_index(['date','stock']))
    Z = X.drop(columns=["price_m", "ret_next_month" ])


    y = X.loc[:,"ret_next_month"]

    Z_train = Z.loc[train_start_date:train_end_date]
    y_train = y.loc[train_start_date:train_end_date]


    Z_valid = Z.loc[valid_start_date:valid_end_date]
    y_valid = y.loc[valid_start_date:valid_end_date]

    return{
        "Z_train": Z_train,
        "y_train": y_train,
        "Z_valid": Z_valid,
        "y_valid": y_valid,
    }




def get_periods(Simulation_start_date,Simulation_end_date,
                train_years = 20,
                valid_years = 1,
                step_years = 1):

    Simulation_start_date=pd.Timestamp(Simulation_start_date)
    Simulation_end_date=pd.Timestamp(Simulation_end_date)


    splits = []

    train_start = Simulation_start_date
    Periods_rows=[]
    max_iter = 50
    i = 0
    while i<max_iter:

        train_end = train_start + pd.DateOffset(years=train_years) - pd.offsets.MonthEnd(1)
        valid_start = train_end + pd.offsets.MonthEnd(1)
        valid_end = valid_start + pd.DateOffset(years=valid_years) - pd.offsets.MonthEnd(1)
        if valid_end > Simulation_end_date:
            break
        Periods_rows.append({
        'train_start': train_start,
        'train_end': train_end,
        'valid_start': valid_start,
        'valid_end': valid_end,
        })

        train_start+= pd.DateOffset(years=step_years)
        i+=1

    Periods= pd.DataFrame(Periods_rows)
    return Periods
















