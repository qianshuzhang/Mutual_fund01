import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

import pyarrow.feather as feather
from statsmodels.formula.api import ols
from scipy import stats


############################################
# part 2
############################################
'''
C & I : Canadian and international 
Bal :   Balanced fund
Bonds : Bonds
Pfd :   Preferred stocks
B & P : Bond and preferred stocks
GS :    Government securities
MM :    Money market fund
TFM :   Tax-free money market fund
'''
crsp_policy_to_exclude = ['C & I', 'Bal', 'Bonds', 'Pfd', 'B & P', 'GS', 'MM', 'TFM']

'''
EIEI : Equity Income Funds
G :    Growth Funds
LCCE : Large-Cap Core Funds
LCGE : Large-Cap Growth Funds
LCVE : Large-Cap Value Funds
MCCE : Mid-Cap Core Funds
MCGE : Mid-Cap Growth Funds
MCVE : Mid-Cap Value Funds
MLCE : Multi-Cap Core Funds
MLGE : Multi-Cap Growth Funds
MLVE : Multi-Cap Value Funds
SCCE : Small-Cap Core Funds
SCGE : Small-Cap Growth Funds
SCVE : Small-Cap Value Funds
'''
lipper_class = ['EIEI', 'G', 'LCCE', 'LCGE', 'LCVE', 'MCCE', 'MCGE', 'MCVE',
                'MLCE', 'MLGE', 'MLVE', 'SCCE', 'SCGE', 'SCVE']
'''
CA : Capital Appreciation Funds
EI : Equity Income Funds
G :  Growth Funds
GI : Growth and Income Funds
MC : Mid-Cap Funds
MR : Micro-Cap Funds
SG : Small-Cap Funds
'''

lipper_obj_cd = ['CA', 'EI', 'G', 'GI', 'MC', 'MR', 'SG']

'''
AGG : Equity USA Aggressive Growth
GMC : Equity USA Midcaps
GRI : Equity USA Growth & Income
GRO : Equity USA Growth
ING : Equity USA Income & Growth
SCG : Equity USA Small Companies
'''
si_obj_cd = ['AGG', 'GMC', 'GRI', 'GRO', 'ING', 'SCG']
'''
G   : Growth
GCI : Growth and current income
IEQ : Equity income
LTG : Long-term growth
MCG : Maximum capital gains
SCG : Small capitalization growth
'''
'''


'''
#wbrger_obj_cd = ['G', 'GCI', 'IEQ', 'LTG', 'MCG', 'SCG']
crsp_policy_to_exclude = ['C & I', 'Bal', 'Bonds', 'Pfd', 'B & P', 'GS', 'MM', 'TFM']
lipper_class = ['EIEI', 'G', 'LCCE', 'LCGE', 'LCVE', 'MCCE', 'MCGE', 'MCVE',
                'MLCE', 'MLGE', 'MLVE', 'SCCE', 'SCGE', 'SCVE']
lipper_obj_cd = ['CA', 'EI', 'G', 'GI', 'MC', 'MR', 'SG']
si_obj_cd = ['AGG', 'GMC', 'GRI', 'GRO', 'ING', 'SCG']
wbrger_obj_cd=['G', 'G-I', 'AGG', 'GCI', 'GRI', 'GRO', 'LTG', 'MCG','SCG']
##############################
#
begin_date='1980-01-01'
end_date='2019-01-31'

##############################
with open('../../data/data/style.feather', 'rb') as f:
    style=feather.read_feather(f)
with open('../../data/data/names.feather', 'rb') as f:
    names=feather.read_feather(f)
with open('../../data/data/fund_summary.feather', 'rb') as f:
    fund_summary=feather.read_feather(f)
names.drop_duplicates(subset=['crsp_fundno'], keep='last', inplace=True)

style=style[(style['begdt'].astype(str)>=begin_date)&(style['enddt'].astype(str)<=end_date)]
names=names[(names['chgdt'].astype(str)>=begin_date)&(names['chgenddt'].astype(str)<=end_date)]
fund_summary=fund_summary[(fund_summary['caldt'].astype(str)>=begin_date)&(fund_summary['caldt'].astype(str)<=end_date)]

# investing on average less than 80% of their assets, excluding cash, in common stocks
fund_summary['per_cash']=fund_summary['per_cash'].replace(np.nan,0)
fund_summary['per_pref']=fund_summary['per_pref'].replace(np.nan,0)
fund_summary['per_eq_oth']=fund_summary['per_eq_oth'].replace(np.nan,0)
fund_summary['per_com']=fund_summary['per_com'].replace(np.nan,0)
fund_summary['per_com_st']=(fund_summary['per_com']+fund_summary['per_pref']+fund_summary['per_eq_oth'])*100/(100-fund_summary['per_cash'])
fund_summary['per_com']=fund_summary['per_com_st']


per_com = fund_summary.groupby(['crsp_fundno']).apply(
    lambda x: np.nansum(x['per_com']) / np.count_nonzero(x['per_com']))
per_com = per_com.reset_index()
per_com.columns = ['crsp_fundno', 'per_com']
style = pd.merge(style, per_com)
style = style[style['per_com'] > 80]

funds = style[(style['lipper_class'].isin(lipper_class)) | (style['lipper_obj_cd'].isin(lipper_obj_cd)) | (
    style['si_obj_cd'].isin(si_obj_cd)) | (style['wbrger_obj_cd'].isin(wbrger_obj_cd))]
funds = funds[~funds['policy'].isin(crsp_policy_to_exclude)]
funds['long_way'] = 1
funds = funds[['crsp_fundno', 'long_way']]
funds.drop_duplicates(inplace=True)
'''
E:Equity
D:Domestic
C Y : Cap-based Style

'''
funds2 = style[(style['crsp_obj_cd'].str[:1] == 'E') & (style['crsp_obj_cd'].str[1:2] == 'D') & (
    style['crsp_obj_cd'].str[2:3].isin(['C', 'Y'])) & (~style['crsp_obj_cd'].str[2:4].isin(['YH', 'YS'])) & (
                       style['si_obj_cd'] != 'OPI')]
funds2['short_way'] = 1
funds2 = funds2[['crsp_fundno', 'short_way']]
funds2.drop_duplicates(inplace=True)

funds4 = pd.merge(funds2, style[['crsp_fundno', 'crsp_obj_cd', 'begdt']], on=['crsp_fundno'],
                  sort=['crsp_fundno', 'begdt'])
funds4['flipper'] = 0
funds4.loc[~funds4['crsp_obj_cd'].str[0:3].isin(['EDC', 'EDY']), 'flipper'] = 1
funds5 = funds4.groupby('crsp_fundno', group_keys=False).apply(lambda x: x.loc[x.flipper.idxmax()])
funds5.reset_index(drop=True, inplace=True)

funds6 = pd.merge(funds5, names, on='crsp_fundno')

counties = funds6.groupby('crsp_fundno')['index_fund_flag'].sum() == 0
funds7 = funds6.loc[counties.values == True]
funds7['namex'] = funds7['fund_name'].str.lower()

funds8 = funds7[(~funds7['namex'].str.contains('index')) & (~funds7['namex'].str.contains('s&p')) & (
    ~funds7['namex'].str.contains('idx')) &
                (~funds7['namex'].str.contains('dfa')) & (~funds7['namex'].str.contains('program')) & (
                    ~funds7['namex'].str.contains('etf')) &
                (~funds7['namex'].str.contains('exchange traded')) & (
                    ~funds7['namex'].str.contains('exchange-traded')) &
                (~funds7['namex'].str.contains('target')) & (~funds7['namex'].str.contains('2005')) &
                (~funds7['namex'].str.contains('2010')) & (~funds7['namex'].str.contains('2015')) &
                (~funds7['namex'].str.contains('2020')) & (~funds7['namex'].str.contains('2025')) &
                (~funds7['namex'].str.contains('2030')) & (~funds7['namex'].str.contains('2035')) &
                (~funds7['namex'].str.contains('2040')) & (~funds7['namex'].str.contains('2045')) &
                (~funds7['namex'].str.contains('2050')) & (~funds7['namex'].str.contains('2055'))]

with open('../../data/data/mflink.feather', 'rb') as f:
    mflink=feather.read_feather(f)

equity_funds = pd.merge(funds8, mflink, on='crsp_fundno')
equity_funds = equity_funds[equity_funds['flipper'] == 0]
'''
wficn_list=equity_funds['wficn'].unique()
holdings=holdings[holdings['wficn'].isin(wficn_list)]

with open('holdings.feather','wb') as f:
    feather.write_feather(holdings,f)
'''
############################################
# part 3
############################################
with open('../../data/data/tna_ret_nav.feather', 'rb') as f:
    tna_ret_nav=feather.read_feather(f)
tna_ret_nav=tna_ret_nav[(tna_ret_nav['caldt'].astype(str)>=begin_date)&(tna_ret_nav['caldt'].astype(str)<=end_date)]

returns1 = pd.merge(equity_funds, tna_ret_nav, on='crsp_fundno')
returns1.rename(columns={'caldt': 'date'}, inplace=True)

# compute gross return
with open('../../data/data/fund_fees.feather', 'rb') as f:
    fund_fees=feather.read_feather(f)
returns2 = pd.merge(returns1, fund_fees, on='crsp_fundno')
returns2 = returns2[(returns2['date'] >= returns2['begdt_y']) & (returns2['date'] <= returns2['enddt'])]
returns2['rret'] = returns2['mret'] + returns2['exp_ratio'] / 12
returns2.sort_values(by=['crsp_fundno', 'date'], inplace=True)

returns2['flag'] = (returns2.crsp_fundno != returns2.crsp_fundno.shift()).astype(int)
returns2['weight'] = returns2['mtna'].shift()
returns2.loc[returns2['flag'] == 1, 'weight'] = returns2.loc[returns2['flag'] == 1, 'mtna']
returns2 = returns2[returns2['weight'] != 0]
returns2['age'] = (returns2['end_dt'] - returns2['first_offer_dt']) / np.timedelta64(1, 'Y')

# keep only observations dated after the fund's first offer date.
returns2 = returns2[returns2['date'] > returns2['first_offer_dt']]

# aggregate multiple share classes
returns2.sort_values(by=['wficn', 'date'], inplace=True)

# monthly return
mret = returns2.groupby(['wficn', 'date']).apply(lambda x: np.average(x['mret'], weights=x['weight']))
returns = mret.reset_index()
returns.columns = ['wficn', 'date', 'mret']

# monthly total net assets
mtna = returns2.groupby(['wficn', 'date']).apply(lambda x: np.sum(x['mtna']))
mtna = mtna.reset_index()
returns['mtna'] = mtna[0]

# weight
weight = returns2.groupby(['wficn', 'date']).apply(lambda x: np.sum(x['weight']))
weight = weight.reset_index()
returns['weight'] = weight[0]

# gross return
rret = returns2.groupby(['wficn', 'date']).apply(lambda x: np.average(x['rret'], weights=x['weight']))
rret = rret.reset_index()
returns['rret'] = rret[0]

# turnover ratio
turnover = returns2.groupby(['wficn', 'date']).apply(lambda x: np.average(x['turn_ratio'], weights=x['weight']))
turnover = turnover.reset_index()
returns['turnover'] = turnover[0]

# expense ratio
expense = returns2.groupby(['wficn', 'date']).apply(lambda x: np.average(x['exp_ratio'], weights=x['weight']))
expense = expense.reset_index()
returns['exp_ratio'] = expense[0]

# fund age
age = returns2.groupby(['wficn', 'date']).apply(lambda x: np.max(x['age']))
age = age.reset_index()
returns['age'] = age[0]

# fund flow
returns['cum_ret'] = (1 + returns['mret']).rolling(12).apply(np.prod, raw=True) - 1
returns['flow'] = (returns['mtna'] - returns.groupby(['wficn'])['mtna'].shift(12) * returns['cum_ret']) / \
                  returns.groupby(['wficn'])['mtna'].shift(12)
returns.loc[returns['flow'] > 20, 'flow'] = np.nan

# fund vol
returns['vol'] = returns.groupby(['wficn'])['mret'].rolling(12).std().reset_index()['mret']

# exclude funds with less than 12 months of observations
obs = returns.groupby(['wficn'])['mtna'].count()
obs = obs.reset_index()
obs.columns = ['wficn', 'obs']
returns = pd.merge(returns, obs)
returns = returns[returns['obs'] >= 12]

# We also exclude fund observations before a fund passes the $5 million threshold for assets under management (AUM).
# All subsequent observations, including those that fall under the $5 million AUM threshold in the future, are included.

mtna_group = 5
returns['tna_ind'] = 0
for w in returns['wficn'].unique():
    for index, row in returns[returns['wficn'] == w].iterrows():
        if row['mtna'] < mtna_group:
            returns.at[index, 'tna_ind'] += 1
        else:
            break

returns.drop(index=returns[returns['tna_ind'] == 1].index, inplace=True)

########### 75% no less than 5 million
aum = returns.sort_values(by=['wficn', 'date'], ascending=False).groupby(['wficn']).head(12)
for index, row in aum.iterrows():
    if row['mtna'] < mtna_group:
        aum.at[index, 'tna_ind'] += 1
total_aum = aum.groupby(['wficn'])['tna_ind'].sum()

returns = returns[~returns['wficn'].isin(total_aum[total_aum > 3].index)]
returns=returns[(returns['date'].astype(str)>=begin_date)&(returns['date'].astype(str)<=end_date)]
################## get summary statistics
'''

with open('returns.feather','rb') as f:
    returns=feather.read_feather(f)
returns=returns[(returns['date'].astype(str)>='1980-01-01')&(returns['date'].astype(str)<'2019-02-01')]
'''
list = ['turnover', 'age', 'flow', 'exp_ratio', 'mtna', 'vol']

Num = []
for i in list:
    Num.append((~returns[i].isna()).sum())

Mean = []
for i in list:
    Mean.append(returns[i].mean())
Mean[1] = returns.groupby(['wficn'])['age'].mean().mean()
Mean[3] = Mean[3] * 100

Std = []
for i in list:
    Std.append(returns[i].std())
Std[1] = returns.groupby(['wficn'])['age'].mean().std()

Med = []
for i in list:
    Med.append(returns[i].median())
Med[1] = returns.groupby(['wficn'])['age'].mean().median()

p10 = []
for i in list:
    p10.append(np.nanpercentile(returns[i], 10))
p10[1] = np.nanpercentile(returns.groupby(['wficn'])['age'].mean(), 10)

p90 = []
for i in list:
    p90.append(np.nanpercentile(returns[i], 90))
p90[1] = np.nanpercentile(returns.groupby(['wficn'])['age'].mean(), 90)

summary = pd.DataFrame([])
for i in ['Num', 'Mean', 'Std', 'Med', 'p10', 'p90']:
    summary[i] = eval(i)
summary.index = list

with open('../../data/data/ff_monthly.feather', 'rb') as f:
    ff_monthly=feather.read_feather(f)
ff_monthly['date'] = ff_monthly['dateff']
ret_ff = pd.merge(returns, ff_monthly, on='date', how='left')
ret_ff['exc_ret'] = ret_ff['mret'] - ret_ff['rf']
ret_ff['Market_adj_ret'] = ret_ff['mret'] - ret_ff['mktrf']
ret_ff = ret_ff[~ret_ff['mret'].isna()]
ret_ff = ret_ff[ret_ff['date'].astype('str') > '2020-01-01']

df_group = ret_ff.groupby('wficn')
reg_result = pd.DataFrame([])

for k, g in df_group:
    # print(k)
    model = ols('exc_ret ~ mktrf', g)
    results = model.fit()
    reg_result.loc[k, 'capm_alpha'] = results.params[0]

    model = ols('exc_ret ~ mktrf+smb+hml', g)
    results = model.fit()
    reg_result.loc[k, 'ff3_alpha'] = results.params[0]

    model = ols('exc_ret ~ mktrf+smb+hml+umd', g)
    results = model.fit()
    reg_result.loc[k, 'ff4_alpha'] = results.params[0]

Num = []
Mean = []
Std = []
Med = []
p10 = []
p90 = []
for i in ['exc_ret', 'Market_adj_ret']:
    Num.append((~ret_ff[i].isna()).sum())

    Mean.append(ret_ff[i].mean())

    Std.append(ret_ff[i].std())

    Med.append(ret_ff[i].median())

    p10.append(np.nanpercentile(ret_ff[i], 10))

    p90.append(np.nanpercentile(ret_ff[i], 90))

for i in ['capm_alpha', 'ff3_alpha', 'ff4_alpha']:
    Num.append((~reg_result[i].isna()).sum())

    Mean.append(reg_result[i].mean())

    Std.append(reg_result[i].std())

    Med.append(reg_result[i].median())

    p10.append(np.nanpercentile(reg_result[i], 10))

    p90.append(np.nanpercentile(reg_result[i], 90))

reg_summary = pd.DataFrame([])
for i in ['Num', 'Mean', 'Std', 'Med', 'p10', 'p90']:
    reg_summary[i] = eval(i)
reg_summary.index = ['exc_ret', 'Market_adj_ret', 'capm_alpha', 'ff3_alpha', 'ff4_alpha']

summary_all = pd.concat([reg_summary, summary])
####

##################### Luck versus Skill in the Cross‐Section of Mutual Fund Returns - FAMA - 2010 ############
with open('../../data/data/ff_monthly.feather', 'rb') as f:
    ff_monthly=feather.read_feather(f)
ff_monthly['date'] = ff_monthly['dateff']
ff_monthly = ff_monthly[
    (ff_monthly['date'].astype('str') >= '1984-01-01') & (ff_monthly['date'].astype('str') <= '2006-09-30')]

ff = []
ff.append(ff_monthly['mktrf'].mean())
ff.append(ff_monthly['smb'].mean())
ff.append(ff_monthly['hml'].mean())
ff.append(ff_monthly['umd'].mean())
ff.append(ff_monthly['mktrf'].std())
ff.append(ff_monthly['smb'].std())
ff.append(ff_monthly['hml'].std())
ff.append(ff_monthly['umd'].std())
ff.append(stats.ttest_1samp(ff_monthly['mktrf'], 0.0)[0])
ff.append(stats.ttest_1samp(ff_monthly['smb'], 0.0)[0])
ff.append(stats.ttest_1samp(ff_monthly['hml'], 0.0)[0])
ff.append(stats.ttest_1samp(ff_monthly['umd'], 0.0)[0])
ff_summary = pd.DataFrame(ff).T
ff_summary.columns = ['mktrf_mean', 'smb_mean', 'hml_mean', 'umd_mean', 'mktrf_std', 'smb_std', 'hml_std', 'umd_std',
                      'mktrf_t', 'smb_t', 'hml_t', 'umd_t']

returns = returns[(returns['date'].astype('str') >= '1984-01-01') & (returns['date'].astype('str') <= '2006-09-30')]

net_ret_ew = returns.groupby(['date'])['mret'].mean().reset_index()
net_ret_ew.columns = ['date', 'net_ret_ew']
ff_monthly = pd.merge(ff_monthly, net_ret_ew)


gross_ret_ew = returns.groupby(['date'])['rret'].mean().reset_index()
gross_ret_ew.columns = ['date', 'gross_ret_ew']
ff_monthly = pd.merge(ff_monthly, gross_ret_ew)


net_ret_vw = returns.groupby(['date']).apply(
    lambda x: np.sum(x['mret'] * x['weight']) / np.sum(x['weight'])).reset_index()
net_ret_vw.columns = ['date', 'net_ret_vw']
ff_monthly = pd.merge(ff_monthly, net_ret_vw)


gross_ret_vw = returns.groupby(['date']).apply(
    lambda x: np.sum(x['rret'] * x['weight']) / np.sum(x['weight'])).reset_index()
gross_ret_vw.columns = ['date', 'gross_ret_vw']
ff_monthly = pd.merge(ff_monthly, gross_ret_vw)

ff_monthly.loc[:,'net_ret_vw']=ff_monthly['net_ret_vw']-ff_monthly['rf']
ff_monthly.loc[:,'gross_ret_vw']=ff_monthly['gross_ret_vw']-ff_monthly['rf']
ff_monthly.loc[:,'net_ret_ew']=ff_monthly['net_ret_ew']-ff_monthly['rf']
ff_monthly.loc[:,'gross_ret_ew']=ff_monthly['gross_ret_ew']-ff_monthly['rf']

Fama_summary_ew = pd.DataFrame(['Intercept', 'mktrf', 'smb', 'hml', 'umd'])
Fama_summary_ew.set_index(0,inplace=True)
Fama_summary_vw = pd.DataFrame(['Intercept', 'mktrf', 'smb', 'hml', 'umd'])
Fama_summary_vw.set_index(0,inplace=True)
# net_ret_ew
del ff_monthly['dateff']
ff_monthly.iloc[:,1:]=ff_monthly.iloc[:,1:]*100

model = ols(formula='net_ret_ew ~ mktrf', data=ff_monthly)
results = model.fit()
Fama_summary_ew['net_ret_ew1'] = results.params
Fama_summary_ew['net_ret_ew1_t'] = results.tvalues
Fama_summary_ew['net_ret_ew1_R2'] = results.rsquared

model = ols('net_ret_ew ~ mktrf+smb+hml', ff_monthly)
results = model.fit()
Fama_summary_ew['net_ret_ew2'] = results.params
Fama_summary_ew['net_ret_ew2_t'] = results.tvalues
Fama_summary_ew['net_ret_ew2_R2'] = results.rsquared

model = ols('net_ret_ew ~ mktrf+smb+hml+umd', ff_monthly)
results = model.fit()
Fama_summary_ew['net_ret_ew3'] = results.params
Fama_summary_ew['net_ret_ew3_t'] = results.tvalues
Fama_summary_ew['net_ret_ew3_R2'] = results.rsquared

# net_ret_vw
model = ols('net_ret_vw ~ mktrf', ff_monthly)
results = model.fit()
Fama_summary_vw['net_ret_vw1'] = results.params
Fama_summary_vw['net_ret_vw1_t'] = results.tvalues
Fama_summary_vw['net_ret_vw1_R2'] = results.rsquared

model = ols('net_ret_vw ~ mktrf+smb+hml', ff_monthly)
results = model.fit()
Fama_summary_vw['net_ret_vw2'] = results.params
Fama_summary_vw['net_ret_vw2_t'] = results.tvalues
Fama_summary_vw['net_ret_vw2_R2'] = results.rsquared

model = ols('net_ret_vw ~ mktrf+smb+hml+umd', ff_monthly)
results = model.fit()
Fama_summary_vw['net_ret_vw3'] = results.params
Fama_summary_vw['net_ret_vw3_t'] = results.tvalues
Fama_summary_vw['net_ret_vw3_R2'] = results.rsquared

# gross_ret_ew
model = ols('gross_ret_ew ~ mktrf', ff_monthly)
results = model.fit()
Fama_summary_ew['gross_ret_ew1'] = results.params
Fama_summary_ew['gross_ret_ew1_t'] = results.tvalues
Fama_summary_ew['gross_ret_ew1_R2'] = results.rsquared

model = ols('gross_ret_ew ~ mktrf+smb+hml', ff_monthly)
results = model.fit()
Fama_summary_ew['gross_ret_ew2'] = results.params
Fama_summary_ew['gross_ret_ew2_t'] = results.tvalues
Fama_summary_ew['gross_ret_ew2_R2'] = results.rsquared

model = ols('gross_ret_ew ~ mktrf+smb+hml+umd', ff_monthly)
results = model.fit()
Fama_summary_ew['gross_ret_ew3'] = results.params
Fama_summary_ew['gross_ret_ew3_t'] = results.tvalues
Fama_summary_ew['gross_ret_ew3_R2'] = results.rsquared

# gross_ret_vw
model = ols('gross_ret_vw ~ mktrf', ff_monthly)
results = model.fit()
Fama_summary_vw['gross_ret_vw1'] = results.params
Fama_summary_vw['gross_ret_vw1_t'] = results.tvalues
Fama_summary_vw['gross_ret_vw1_R2'] = results.rsquared

model = ols('gross_ret_vw ~ mktrf+smb+hml', ff_monthly)
results = model.fit()
Fama_summary_vw['gross_ret_vw2'] = results.params
Fama_summary_vw['gross_ret_vw2_t'] = results.tvalues
Fama_summary_vw['gross_ret_vw2_R2'] = results.rsquared

model = ols('gross_ret_vw ~ mktrf+smb+hml+umd', ff_monthly)
results = model.fit()
Fama_summary_vw['gross_ret_vw3'] = results.params
Fama_summary_vw['gross_ret_vw3_t'] = results.tvalues
Fama_summary_vw['gross_ret_vw3_R2'] = results.rsquared

Fama_summary_ew = Fama_summary_ew.T
Fama_summary_vw = Fama_summary_vw.T

scale=1
Fama_summary_ew.loc['net_ret_ew1','Intercept']=Fama_summary_ew.loc['net_ret_ew1','Intercept']*12*scale
Fama_summary_ew.loc['net_ret_ew2','Intercept']=Fama_summary_ew.loc['net_ret_ew2','Intercept']*12*scale
Fama_summary_ew.loc['net_ret_ew3','Intercept']=Fama_summary_ew.loc['net_ret_ew3','Intercept']*12*scale

Fama_summary_vw.loc['net_ret_vw1','Intercept']=Fama_summary_vw.loc['net_ret_vw1','Intercept']*12*scale
Fama_summary_vw.loc['net_ret_vw2','Intercept']=Fama_summary_vw.loc['net_ret_vw2','Intercept']*12*scale
Fama_summary_vw.loc['net_ret_vw3','Intercept']=Fama_summary_vw.loc['net_ret_vw3','Intercept']*12*scale

Fama_summary_ew.loc['gross_ret_ew1','Intercept']=Fama_summary_ew.loc['gross_ret_ew1','Intercept']*12*scale
Fama_summary_ew.loc['gross_ret_ew2','Intercept']=Fama_summary_ew.loc['gross_ret_ew2','Intercept']*12*scale
Fama_summary_ew.loc['gross_ret_ew3','Intercept']=Fama_summary_ew.loc['gross_ret_ew3','Intercept']*12*scale

Fama_summary_vw.loc['gross_ret_vw1','Intercept']=Fama_summary_vw.loc['gross_ret_vw1','Intercept']*12*scale
Fama_summary_vw.loc['gross_ret_vw2','Intercept']=Fama_summary_vw.loc['gross_ret_vw2','Intercept']*12*scale
Fama_summary_vw.loc['gross_ret_vw3','Intercept']=Fama_summary_vw.loc['gross_ret_vw3','Intercept']*12*scale





