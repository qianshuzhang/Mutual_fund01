import pandas as pd
import numpy as np
import datetime as dt
import pyarrow.feather as feather

with open('/home/qianshu/US_factor/chars60_rank_imputed.feather','rb') as f:
    char=feather.read_feather(f)
with open('holdings.feather','rb') as f:
   holdings=feather.read_feather(f)

char['date']=char['date'].astype('str')
char['month']=char['date'].apply(lambda x:x[0:4] + '-' + x[5:7])
holdings['month']=holdings['month'].astype('str')
a=pd.merge(char,holdings,on=['month','permno'])

char_list=[
       'lag_me', 'rank_ill', 'rank_me_ia', 'rank_chtx', 'rank_mom36m',
       'rank_re', 'rank_depr', 'rank_rd_sale', 'rank_roa', 'rank_bm_ia',
       'rank_cfp', 'rank_mom1m', 'rank_baspread', 'rank_rdm', 'rank_bm',
       'rank_sgr', 'rank_mom12m', 'rank_std_dolvol', 'rank_rvar_ff3',
       'rank_herf', 'rank_sp', 'rank_hire', 'rank_pctacc', 'rank_grltnoa',
       'rank_turn', 'rank_abr', 'rank_seas1a', 'rank_adm', 'rank_me',
       'rank_cash', 'rank_chpm', 'rank_cinvest', 'rank_acc', 'rank_gma',
       'rank_beta', 'rank_sue', 'rank_cashdebt', 'rank_ep', 'rank_lev',
       'rank_op', 'rank_alm', 'rank_lgr', 'rank_noa', 'rank_roe',
       'rank_dolvol', 'rank_rsup', 'rank_std_turn', 'rank_maxret',
       'rank_mom6m', 'rank_ni', 'rank_nincr', 'rank_ato', 'rank_rna',
       'rank_agr', 'rank_zerotrade', 'rank_chcsho', 'rank_dy',
       'rank_rvar_capm', 'rank_rvar_mean', 'rank_mom60m', 'rank_pscore',
       'rank_pm', 'log_me']
g = a.groupby(['wficn', 'month']).apply(lambda x: np.average(x['log_me'], weights=x['D']))
fund_char=g.reset_index()
fund_char.columns=['wficn','month','log_me']
for i in char_list:
    g = a.groupby(['wficn', 'month']).apply(lambda x: np.average(x[i], weights=x['D']))
    fund_char[i]=g.reset_index()[0]
    print(i+' finished')

with open('fund_char1.feather','wb') as f:
    feather.write_feather(fund_char,f)

