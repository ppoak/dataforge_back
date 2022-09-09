import os 
os.chdir('/root/Quant')

import pandas as pd
import akshare as ak
from functools import partial
from joblib import Parallel, delayed
from collectors.libs import AkShare, format_code


if __name__ == "__main__":
    # benchmark: 23m+
    format_code_partial = partial(format_code, format_str="{market}{code}")
    codes = list(map(format_code_partial, ak.stock_zh_a_spot_em()['代码'].to_list()))
    # comment the `akshare/stock_feature/stock_three_report_em.py:56` tqdm part
    joblibres = Parallel(n_jobs=12, backend='loky')(delayed(AkShare.balance_sheet)(code) for code in codes)
    data = pd.concat(joblibres)
    data = data.sort_index()
    data.to_parquet('../data/financials/balance_sheet.parquet', compression='gzip')

    # benchmark: 18m+
    format_code_partial = partial(format_code, format_str="{market}{code}")
    codes = list(map(format_code_partial, ak.stock_zh_a_spot_em()['代码'].to_list()))
    # comment the `akshare/stock_feature/stock_three_report_em.py:56` tqdm part
    joblibres = Parallel(n_jobs=12, backend='loky')(delayed(AkShare.profit_sheet)(code) for code in codes)
    data = pd.concat(joblibres)
    data = data.sort_index()
    data.to_parquet('../data/financials/profit_sheet.parquet', compression='gzip')

    # benchmark: 18m+
    format_code_partial = partial(format_code, format_str="{market}{code}")
    codes = list(map(format_code_partial, ak.stock_zh_a_spot_em()['代码'].to_list()))
    # comment the `akshare/stock_feature/stock_three_report_em.py:56` tqdm part
    joblibres = Parallel(n_jobs=12, backend='loky')(delayed(AkShare.cashflow_sheet)(code) for code in codes)
    data = pd.concat(joblibres)
    data = data.sort_index()
    data.to_parquet('../data/financials/cashflow_sheet.parquet', compression='gzip')