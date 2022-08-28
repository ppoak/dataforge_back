import re
import time
import random
import requests
import pandas as pd
import akshare as ak
from functools import partial


def get_stock_code():
    codes = ak.stock_zh_a_spot_em()['代码'].sort_values().tolist()
    func = partial(format_code, formatstr='{market}{code}')
    codes = list(map(func, codes))
    return codes

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
        time.sleep(0.8)
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
