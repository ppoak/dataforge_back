import time
import random
import requests
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
