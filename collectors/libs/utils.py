import re
import time
import random
import requests
import numpy as np
import pandas as pd


def get_proxy(page_size: int = 20):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36"
    }
    url_list = [f'https://free.kuaidaili.com/free/inha/{i}/' for i in range(1, page_size + 1)]
    proxies = []
    for url in url_list:
        data = pd.read_html(url)[0][['IP', 'PORT', '类型']].drop_duplicates()
        print(f'[+] {url} Get Success!')
        data['类型'] = data['类型'].str.lower()
        proxy = (data['类型'] + '://' + data['IP'] + ':' + data['PORT'].astype('str')).to_list()
        proxies += list(map(lambda x: {x.split('://')[0]: x}, proxy))
        time.sleep(1.2)
    available_proxies = []
    
    for proxy in proxies:
        try:
            res = requests.get('https://www.baidu.com', 
                headers=headers, proxies=proxy, timeout=1)
            res.raise_for_status()
            available_proxies.append(proxy)
        except Exception as e:
            print(str(e))
    
    print(f'[=] Get {len(proxies)} proxies, while {len(available_proxies)} are available. '
        f'Current available rate is {len(available_proxies) / len(proxies) * 100:.2f}%')
    return proxies

def proxy_request(
    url: str, 
    proxies: 'dict | list', 
    retry: int = None, 
    timeout: int = 1,
    delay: int = 0,
    verbose: bool = True,
    **kwargs
):
    if isinstance(proxies, dict):
        proxies = [proxies]
    retry = retry or len(proxies)
    random.shuffle(proxies) 
    for try_times, proxy in enumerate(proxies):
        if try_times + 1 <= retry:
            try:
                response = requests.get(url, proxies=proxy, timeout=timeout, **kwargs)
                response.raise_for_status()
                if verbose:
                    print(f'[+] {url}, try {try_times + 1}/{retry}')
                return response
            except Exception as e:
                if verbose:
                    print(f'[-] [{e}] {url}, try {try_times + 1}/{retry}')
                time.sleep(delay)

def chinese_holidays():
    root = 'https://api.apihubs.cn/holiday/get'
    complete = False
    page = 1
    holidays = []
    while not complete:
        params = f'?field=date&holiday_recess=1&cn=1&page={page}&size=366'
        url = root + params
        data = requests.get(url, verbose=False).get().json['data']
        if data['page'] * data['size'] >= data['total']:
            complete = True
        days = pd.DataFrame(data['list']).date.astype('str')\
            .astype('datetime64[ns]').to_list()
        holidays += days
        page += 1
    return 

def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def strip_stock_code(code: str):
    code_pattern = r'\.?[Ss][Zz]\.?|\.?[Ss][Hh]\.?|\.?[Bb][Jj]\.?'\
        '|\.?[Oo][Ff]\.?'
    return re.sub(code_pattern, '', code)

def format_code(code, format_str = '{market}.{code}'):
    if len(c := code.split('.')) == 2:
        dig_code = c.pop(0 if c[0].isdigit() else 1)
        market_code = c[0]
        return format_str.format(market=market_code, code=dig_code)
    elif len(code.split('.')) == 1:
        sh_code_pat = '6\d{5}|9\d{5}'
        sz_code_pat = '0\d{5}|2\d{5}|3\d{5}'
        bj_code_pat = '8\d{5}|4\d{5}'
        if re.match(sh_code_pat, code):
            return format_str.format(code=code, market='sh')
        if re.match(sz_code_pat, code):
            return format_str.format(code=code, market='sz')
        if re.match(bj_code_pat, code):
            return format_str.format(code=code, market='bj')
    else:
        raise ValueError("Your input code is not unstood")

