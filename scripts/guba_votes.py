import datetime
import requests
import pandas as pd
import akshare as ak
from joblib import Parallel, delayed

import sys
sys.path.append('.')
import os
os.chdir("/root/Quant")
from collectors.libs.utils import format_code


def crawl_stock(code: str):
    today = datetime.datetime.today().date()
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15",
        "Referer": "http://guba.eastmoney.com/",
        "Host": "gubacdn.dfcfw.com"
    }
    code = format_code(code, '{market}{code}')
    url = f"http://gubacdn.dfcfw.com/LookUpAndDown/{code}.js"
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    res = eval(res.text.strip('var LookUpAndDown=').replace('null', f'"{today}"'))
    data = pd.Series(res['Data'])
    data['code'] = code
    return data

# benchmark: 36s
today = datetime.datetime.today().strftime('%Y%m%d')
codes = ak.stock_zh_a_spot_em()['代码'].to_list()
datas = Parallel(n_jobs=-1, backend='threading')(delayed(crawl_stock)(code) for code in codes)
data = pd.concat(datas, axis=1).T
data = data.set_index('code').drop('Date', axis=1)
data = data.astype({"TapeZ": "float32", "TapeD": "float32", "TapeType": "uint8"})
data = pd.concat([data], keys=[pd.to_datetime(today)], names=['datetime', 'instrument'])
data.to_parquet(f'./data/derivative_indicators/guba_votes/{today}.parquet')