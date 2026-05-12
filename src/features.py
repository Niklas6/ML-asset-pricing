

import pandas as pd

import numpy as np
from IPython.core.display_functions import display



def generate_features(train_start_date,test_end_date):
    df = pd.read_csv('../data/raw/prices.csv', index_col='Date', parse_dates=True)
    #ts = pd.Timestamp(train_start_date)
    #data_load_time = ts - pd.DateOffset(years=5)
    #data_load_time=data_load_time.strftime("%Y-%m-%d")
    df = df.loc[train_start_date:test_end_date]




    price_monthly = df.resample("ME").mean()
    valid_names =(price_monthly.iloc[0]!=0) & (price_monthly.iloc[0].notna()) & (price_monthly.iloc[0].notnull())
    price_monthly=price_monthly.loc[:,valid_names]

    #df.resample("ME").mean()
    #df.rolling(21).mean()

    #exists= (price_monthly >0.1).astype(int)
    #exists_5y=exists.copy()
    #exists_5y.index = exists_5y.index - pd.DateOffset(years=5)
    ret_1m=price_monthly.pct_change(1)
    ret_3m=price_monthly.pct_change(3)
    ret_6m=price_monthly.pct_change(6)
    ret_12m=price_monthly.pct_change(12)

    vol_1m = df.resample("ME").std()
    vol_1m= vol_1m.loc[:,valid_names]
    vol_3m = vol_1m.rolling(window=3).mean()
    vol_6m = vol_1m.rolling(window=6).mean()
    vol_12m = vol_1m.rolling(window=12).mean()
    ret_next_month = ret_1m.shift(-1)



    averages = pd.concat(
        {
            "price_m": price_monthly,
            #'exists': exists,
            "ret_m1": ret_1m,
            "ret_m3": ret_3m,
            "ret_m6": ret_6m,
            "ret_y1": ret_12m,
            'vol_m1': vol_1m,
            'vol_m3': vol_3m,
            'vol_m6': vol_6m,
            'vol_y1': vol_12m,
            'ret_next_month': ret_next_month,
        },
        axis=1
    )

    averages = averages.swaplevel(axis=1).sort_index(axis=1)

    averages.index.name = "date"
    averages.columns.names = ["stock", "feature"]
    averages = (
        averages
        .stack(level="stock")
        .reset_index()
    )
    #averages = averages.sort_values(["date", "stock"])
    feature_cols = ["price_m","ret_m1", "ret_m3", "ret_m6", "ret_y1", 'vol_m1', 'vol_m3', 'vol_m6', 'vol_y1','ret_next_month']
    averages = averages.replace([np.inf, -np.inf], np.nan)
    averages = averages.dropna(subset=feature_cols)
    averages = averages.set_index(['date',	'stock'])


    return averages




def get_dataset(period):
    train_start_date=period['train_start'].strftime("%Y-%m-%d")
    train_end_date=period['train_end'].strftime("%Y-%m-%d")
    valid_start_date=period['valid_start'].strftime("%Y-%m-%d")
    valid_end_date=period['valid_end'].strftime("%Y-%m-%d")
    #test_start_date=period['test_start'].strftime("%Y-%m-%d")
    #test_end_date=period['test_end'].strftime("%Y-%m-%d")
    #Z2 = pd.read_csv('../data/processed/Z.csv', index_col='Date', parse_dates=True)
    #Z = Z.loc[train_start_date,test_end_date]
    X = generate_features(train_start_date, valid_end_date)
    Z = X.drop(columns=["price_m", "ret_next_month" ])


    y = X.loc[:,"ret_next_month"]
    #display(Z)
    #display(Z)
    #display(Z.xs( ("ret_m1",'exists'),  level=1, axis=1).shift(-1).fillna(0))



    Z = Z.iloc[:-1]
    y = y.iloc[:-1]

    Z_train = Z.loc[train_start_date:train_end_date]
    y_train = y.loc[train_start_date:train_end_date]


    Z_valid = Z.loc[valid_start_date:valid_end_date]
    y_valid = y.loc[valid_start_date:valid_end_date]

    #Z_test = Z.loc[test_start_date:test_end_date]
    #y_test = y.loc[test_start_date:test_end_date]
    return{
        "Z_train": Z_train,
        "y_train": y_train,
        "Z_valid": Z_valid,
        "y_valid": y_valid,
        #"Z_test": Z_test,
        #"y_test": y_test,
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
















