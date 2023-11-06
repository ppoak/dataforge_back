import re
import time
import requests
import pandas as pd
import quantframe as qf
from joblib import Parallel, delayed


class KaiXin(qf.RequestBase):

    def __init__(self, page_count: int = 10):
        url = [f"http://www.kxdaili.com/dailiip/2/{i}.html" for i in range(1, page_count + 1)]
        super().__init__(url = url)
    
    def callback(self):
        results = []
        etrees = self.etree
        for tree in etrees:
            if tree is None:
                continue
            for tr in tree.xpath("//table[@class='active']//tr")[1:]:
                ip = "".join(tr.xpath('./td[1]/text()')).strip()
                port = "".join(tr.xpath('./td[2]/text()')).strip()
                results.append({
                    "http": "http://" + "%s:%s" % (ip, port),
                    "https": "https://" + "%s:%s" % (ip, port)
                })
        return pd.DataFrame(results)


class KuaiDaili(qf.RequestBase):

    def __init__(self, page_count: int = 20):
        url_pattern = [
            'https://www.kuaidaili.com/free/inha/{}/',
            'https://www.kuaidaili.com/free/intr/{}/'
        ]
        url = []
        for page_index in range(1, page_count + 1):
            for pattern in url_pattern:
                url.append(pattern.format(page_index))
        super().__init__(url=url, delay=4)

    def callback(self):
        results = []
        for tree in self.etree:
            if tree is None:
                continue
            proxy_list = tree.xpath('.//table//tr')
            for tr in proxy_list[1:]:
                results.append({
                    "http": "http://" + ':'.join(tr.xpath('./td/text()')[0:2]),
                    "https": "http://" + ':'.join(tr.xpath('./td/text()')[0:2])
                })
        return pd.DataFrame(results)


class Ip3366(qf.RequestBase):

    def __init__(self, page_count: int = 3):
        url = []
        url_pattern = ['http://www.ip3366.net/free/?stype=1&page={}', "http://www.ip3366.net/free/?stype=2&page={}"]
        for page in range(1, page_count + 1):
            for pat in url_pattern:
                url.append(pat.format(page))
        super().__init__(url=url)

    def callback(self, *args, **kwargs):
        results = []
        for text in self.html:
            if text is None:
                continue
            proxies = re.findall(r'<td>(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td>[\s\S]*?<td>(\d+)</td>', text)
            for proxy in proxies:
                results.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return pd.DataFrame(results)


class Ip98(qf.RequestBase):

    def __init__(self, page_count: int = 20):
        super().__init__(url=[f"https://www.89ip.cn/index_{i}.html" for i in range(1, page_count + 1)])
    
    def callback(self, *args, **kwargs):
        results = []
        for text in self.html:
            if text is None:
                continue
            proxies = re.findall(
                r'<td.*?>[\s\S]*?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\s\S]*?</td>[\s\S]*?<td.*?>[\s\S]*?(\d+)[\s\S]*?</td>',
                text
            )
            for proxy in proxies:
                results.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return pd.DataFrame(results)


class Checker(qf.RequestBase):

    def __init__(self, proxies: list[dict]):
        super().__init__(
            url = ["http://httpbin.org/ip"] * len(proxies),
            proxies = proxies,
            timeout = 2.0,
            retry = 1,
        )

    def _req(self, url: str, proxy: dict):
        method = getattr(requests, self.method)
        retry = self.retry or 1
        for t in range(1, retry + 1):
            try:
                resp = method(
                    url, headers=self.headers, proxies=proxy,
                    timeout=self.timeout, **self.kwargs
                )
                resp.raise_for_status()
                if self.verbose:
                    print(f'[+] {url} try {t}')
                return resp
            except Exception as e:
                if self.verbose:
                    print(f'[-] {e} {url} try {t}')
                time.sleep(self.delay)
        return None

    def request(self) -> list[requests.Response]:
        responses = []
        for proxy, url in zip(self.proxies, self.url):
            resp = self._req(url=url, proxy=proxy)
            responses.append(resp)
        self.responses = responses
        return self
    
    def para_request(self) -> list[requests.Response]:
        self.responses = Parallel(n_jobs=-1, backend='loky')(
            delayed(self._req)(url, proxy) for url, proxy in zip(self.url, self.proxies)
        )
        return self
    
    def callback(self):
        results = []
        for i, res in enumerate(self.responses):
            if res is None:
                continue
            results.append(self.proxies[i])
        return pd.DataFrame(results)
                


if __name__ == "__main__":
    from database import Database

    print(f'[=] Loading other database and read proxy data ...')
    otherdb = Database('/home/kali/data/other')
    if otherdb.config.get('proxy'):
        proxy = otherdb.load('proxy')
        proxy = proxy['proxy']
    else:
        proxy = None

    print(f'[=] Fetching kaixin proxy source ...')
    kx = KaiXin()()
    print(f'[=] Fetching kuaidaili proxy source ...')
    kdl = KuaiDaili()(para=False)
    print(f'[=] Fetching ip3366 proxy source ...')
    ip3366 = Ip3366()()
    print(f'[=] Fetching ip98 proxy source ...')
    ip98 = Ip98()()

    print(f'[=] Checking availability or proxies ...')
    data = pd.concat([proxy, kx, kdl, ip3366, ip98], ignore_index=True)
    data = data.to_dict(orient='records')
    res = Checker(data)()

    print(f'[=] Dumping data into other database table proxy ...')
    res.index = res.index.map(lambda x: str(x).zfill(6))
    otherdb = otherdb.dump(res, 'proxy')

    print(f'[+] Update Success!')
    print(otherdb)
