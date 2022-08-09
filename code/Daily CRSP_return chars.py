import pandas as pd
import numpy as np
# pd.set_option('display.max_columns', None)
import random
import statsmodels.api as sm

# import EPFR fund IDs
# ID = pd.read_csv('FundID_CUSIP_aug26.csv', index_col=0)

# CUSIPS = pd.unique(ID['CUSIP'].apply(lambda x: x[:-1]))

# import CRSP data
CRSP = pd.read_csv('CRSP_return 2009-2021.csv')

# keep the CRSP data that is also in EPFR
# CRSP = CRSP[CRSP['cusip8'].isin(CUSIPS)]

# date
CRSP['caldt'] = pd.to_datetime(CRSP['caldt'], format='%Y%m%d')

del CRSP ['crsp_fundno']
del CRSP ['dnav']

CRSP = CRSP.rename(columns={'cusip8':'FundName','caldt':'ReportDate','dret':'return'})
CRSP = CRSP.reset_index(drop=True)

# the return column has 'R'
CRSP['return'] = CRSP['return'].replace('R',np.nan)
CRSP['return'] = CRSP['return'].astype('float')

# do sorting on FundName and date firstly
CRSP['return'] = CRSP['return'].fillna(method='ffill')

Data = CRSP

# merge with external factor data
Factor = pd.read_csv('Factors4.csv',index_col=0)
Factor['Date'] = pd.to_datetime(Factor['Date'])
Data = pd.merge(Data, Factor, how='left', left_on='ReportDate', right_on='Date')
Data = Data.dropna(axis=0)
Data = Data.reset_index(drop=True)

# compute excess return
Data['excess return'] = Data['return'] - Data['RF']

################################################################################################################
# STR
Data['STR'] = Data[['FundName','excess return']].groupby('FundName').shift(1)

# MOM_weekly
Data['MOM5'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=7,min_periods=7).apply(lambda x: np.sum(np.log(pd.Series(x).iloc[1:6]/100+1))).reset_index(drop=True)*100

# MOM_monthly
Data['MOM22'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=20,min_periods=20).apply(lambda x: np.sum(np.log(pd.Series(x).iloc[6:23]/100+1))).reset_index(drop=True)*100

# MOM_half-year
Data['MOM120'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=100,min_periods=100).apply(lambda x: np.sum(np.log(pd.Series(x).iloc[23:121]/100+1))).reset_index(drop=True)*100

# MOM_year
Data['MOM240'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=120,min_periods=120).apply(lambda x: np.sum(np.log(pd.Series(x).iloc[121:241]/100+1))).reset_index(drop=True)*100

#print('MOM')

# Value at risk
Data['VaR5'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=60,min_periods=60).apply(lambda x: pd.Series(x).nsmallest(3).iloc[2]).reset_index(drop=True)
Data['VaR10'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=60,min_periods=60).apply(lambda x: pd.Series(x).nsmallest(6).iloc[5]).reset_index(drop=True)

print('VaR')

# Expected shortfall
Data['ES5'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=60,min_periods=60).apply(lambda x: pd.Series(x).nsmallest(3).mean()).reset_index(drop=True)
Data['ES10'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=60,min_periods=60).apply(lambda x: pd.Series(x).nsmallest(6).mean()).reset_index(drop=True)

#print('ES')

# Moments
Data['VOL'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=60,min_periods=60).std().reset_index(drop=True)
Data['SKEW'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=60,min_periods=60).skew().reset_index(drop=True)
Data['KURT'] = Data[['FundName','excess return']].groupby('FundName').rolling(window=60,min_periods=60).kurt().reset_index(drop=True)

#print('Moments')

# define the factors
CAPM = ['Mkt-RF']
FF3 = ['Mkt-RF','SMB','HML']
C4 = ['Mkt-RF','SMB','HML','Mom']

# Alpha
def fundalpha(data,factor):
    alpha = []
    names = pd.unique(data['FundName'])
    for name in names:
        temp_data = data[data['FundName']==name]
        for i in range(len(temp_data)):
            if i<60:
                alpha.append(np.nan)
            else:
                y = data['excess return'].iloc[i-60:i]
                x = data[factor].iloc[i-60:i,:]
                results = sm.OLS(y/100,sm.add_constant(x/100)).fit()
                alpha.append(results.params[0])
    return alpha            


Data['ALPHA_CAPM'] = fundalpha(Data,CAPM)
Data['ALPHA_FF3'] = fundalpha(Data,FF3)
Data['ALPHA_C4'] = fundalpha(Data,C4)

# print('alpha')

# Variance
def fundvar(data,factor):
    var = []
    names = pd.unique(data['FundName'])
    for name in names:
        temp_data = data[data['FundName']==name]
        for i in range(len(temp_data)):
            if i<60:
                var.append(np.nan)
            else:
                y = data['excess return'].iloc[i-60:i]
                x = data[factor].iloc[i-60:i,:]
                results = sm.OLS(y/100,sm.add_constant(x/100)).fit()
                var.append(np.var(results.resid))
    return var

Data['VAR_CAPM'] = fundvar(Data,CAPM)
Data['VAR_FF3'] = fundvar(Data,FF3)
Data['VAR_C4'] = fundvar(Data,C4)

# print('Var')

Data.to_csv('CRSP_return.csv')