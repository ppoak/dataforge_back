import numpy as np
import pandas as pd
from pathlib import Path
from libs import compute_ic, rebalance


exp_name = 'lgbm_1d/oos'

# read result
path = Path(f'./data/intermediate/results/{exp_name}')
pred = pd.read_parquet(path)
pred = pred.loc[~pred.index.duplicated(keep='last')]

# read return data and benchmark data
ret = pd.read_parquet('./data/intermediate/forward_return/1d_open_open.parquet')
benchmark = pd.read_parquet('./data/index/zz500.parquet', columns=['ret'])
benchmark = benchmark.loc[pred.index.levels[0]]

# construct pred_label variable
pred_label = pd.concat([pred, ret], axis=1, join='inner')

# backtest
result = rebalance(pred_label, benchmark=benchmark, N=10, commission_ratio=0.001)
ic = compute_ic(pred_label)['ic']
ic_mean = ic.rolling(20).mean()

# result saving
with pd.ExcelWriter(f'data/intermediate/results/{exp_name.split("/")[0]}_sheets.xlsx') as writer:
    result['profit'].to_excel(writer, sheet_name='profit')
    (result['profit'] + 1).cumprod().to_excel(writer, sheet_name='cumprofit')
    result['exprofit'].to_excel(writer, sheet_name='exprofit')
    (result['exprofit'] + 1).cumprod().to_excel(writer, sheet_name='excumprofit')
    result['mmd'].to_excel(writer, sheet_name='maxdrawdown')
    result['exmmd'].to_excel(writer, sheet_name='exmaxdrawdown')
    ic.to_excel(writer, sheet_name='ic')
    ic_mean.to_excel(writer, sheet_name='icmean')
