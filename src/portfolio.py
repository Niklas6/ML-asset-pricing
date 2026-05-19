
from dataclasses import dataclass


import pandas as pd




@dataclass(slots=True)
class Portfolio:
    long: pd.DataFrame
    short: pd.DataFrame

def uniform_25(y_pred_month):
    n = int(len(y_pred_month) / 4)
    long = y_pred_month.tail(n)
    short = y_pred_month.head(n)
    long['weight'] = 1 / n
    short['weight'] = 1 / n

    port = Portfolio(long=long, short=short)
    return port

def portfolio_weight(y_pred_month):
    #n = int(len(y_pred_month) / 2)

    rel_pred= y_pred_month-y_pred_month.mean()


    long = y_pred_month.loc[rel_pred['prediction']>0]
    short = y_pred_month.loc[rel_pred['prediction']<0]

    rel_long = rel_pred.loc[rel_pred['prediction']>0]
    rel_short = rel_pred.loc[rel_pred['prediction']<0]



    long['weight'] = rel_long/rel_long.sum()/2
    short['weight'] =rel_short/rel_short.sum()/2

    port = Portfolio(long=long, short=short)
    return port


def backtest_diag(df):
    #arithmetic_average_return= df.mean()
    geometric_average_return= ((1 + df).prod() ** (1 / len(df))) - 1
    volatility= df.std()
    max_drawdown= df.min()



    dg=pd.DataFrame({
        'period':str(df.index[0])+'-'+str(df.index[-1]),
        'annual return': geometric_average_return,
        'volatility':volatility,
        'sharpe': geometric_average_return/volatility,
        'max drawdown': max_drawdown
    })
    return dg


def revenue_pred(port, y_valid_month):

    return(y_valid_month.loc[port.long.index]*port.long.loc[:,'weight']).sum()-(y_valid_month.loc[port.short.index]*port.short.loc[:,'weight']).sum()