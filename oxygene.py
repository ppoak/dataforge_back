import re
import time
import requests
import numpy as np
import pandas as pd
import quantframe as qf
from lxml import etree
from pathlib import Path
from joblib import Parallel, delayed


def format_code(code, format_str = '{market}.{code}', upper: bool = True):
    if len(c := code.split('.')) == 2:
        dig_code = c.pop(0 if c[0].isdigit() else 1)
        market_code = c[0]
        if upper:
            market_code = market_code.upper()
        return format_str.format(market=market_code, code=dig_code)
    elif len(code.split('.')) == 1:
        sh_code_pat = '6\d{5}|9\d{5}'
        sz_code_pat = '0\d{5}|2\d{5}|3\d{5}'
        bj_code_pat = '8\d{5}|4\d{5}'
        if re.match(sh_code_pat, code):
            return format_str.format(code=code, market='sh' if not upper else 'SH')
        if re.match(sz_code_pat, code):
            return format_str.format(code=code, market='sz' if not upper else 'SZ')
        if re.match(bj_code_pat, code):
            return format_str.format(code=code, market='bj' if not upper else 'BJ')
    else:
        raise ValueError("Your input code is not unstood")


class ProxyFetcher:

    headers = {
        "User-Agent": 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101',
    }

    def proxy_kaixin(self, page_count: int = 10):
        result = []

        target_urls = [f"http://www.kxdaili.com/dailiip/2/{i}.html" for i in range(1, page_count + 1)]
        for url in target_urls:
            tree = etree.HTML(requests.get(url, headers=self.headers).text)
            for tr in tree.xpath("//table[@class='active']//tr")[1:]:
                ip = "".join(tr.xpath('./td[1]/text()')).strip()
                port = "".join(tr.xpath('./td[2]/text()')).strip()
                result.append({"http": "http://" + "%s:%s" % (ip, port),
                            "https": "https://" + "%s:%s" % (ip, port)})
        return result

    def proxy_kuaidaili(self, page_count: int = 20):
        result = []

        url_pattern = [
            'https://www.kuaidaili.com/free/inha/{}/',
            'https://www.kuaidaili.com/free/intr/{}/'
        ]
        url_list = []
        for page_index in range(1, page_count + 1):
            for pattern in url_pattern:
                url_list.append(pattern.format(page_index))
                
        for url in url_list:
            tree = etree.HTML(requests.get(url, headers=self.headers).text)
            proxy_list = tree.xpath('.//table//tr')
            time.sleep(1)
            for tr in proxy_list[1:]:
                result.append({
                    "http": "http://" + ':'.join(tr.xpath('./td/text()')[0:2]),
                    "https": "http://" + ':'.join(tr.xpath('./td/text()')[0:2])
                })
        return result

    def proxy_ip3366(self, page_count: int = 3):
        result = []
        urls = ['http://www.ip3366.net/free/?stype=1&page={}', "http://www.ip3366.net/free/?stype=2&page={}"]
        url_list = []

        for page in range(1, page_count + 1):
            for url in urls:
                url_list.append(url.format(page))

        for url in url_list:
            r = requests.get(url, headers=self.headers)
            proxies = re.findall(r'<td>(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td>[\s\S]*?<td>(\d+)</td>', r.text)
            for proxy in proxies:
                result.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return result

    def proxy_89ip(self, page_count: int = 20):
        result = []
        urls = [f"https://www.89ip.cn/index_{i}.html" for i in range(1, page_count + 1)]
        for url in urls:
            r = requests.get(url, headers=self.headers, timeout=10)
            proxies = re.findall(
                r'<td.*?>[\s\S]*?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\s\S]*?</td>[\s\S]*?<td.*?>[\s\S]*?(\d+)[\s\S]*?</td>',
                r.text)
            for proxy in proxies:
                result.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return result
    
    def _check(
        self, 
        proxy: dict, 
        url: str = "http://www.baidu.com",
        retry: float = 1, 
        timeout: float = 1,
        delay: float = 0
    ) -> bool:
        for t in range(retry):
            try:
                resp = requests.get(url, timeout=timeout, proxies=proxy)
                resp.raise_for_status()
                return True
            except Exception as e:
                time.sleep(delay)
        return False
    
    def run(self):
        all_funcs = filter(lambda x: not x.startswith('_') and x != "run" and x != "headers", dir(self))
        proxies = Parallel(n_jobs=-1, backend='loky')(
            delayed(getattr(self, func))() for func in all_funcs
        )
        proxies = sum(proxies, [])

        results = np.array(Parallel(n_jobs=-1, backend='loky')(
            delayed(self._check)(
                proxy = proxy,
            ) for proxy in proxies
        ))
        
        df = pd.DataFrame(np.array(proxies)[results == True].tolist())
        return df


class Database(qf.DatabaseBase):

    def __init__(
        self, path: str,
        size_limit: int = 100 * 2 ** 20,
        item_limit: int = 1e10,
    ) -> None:
        self.size_limit = int(size_limit)
        self.item_limit = int(item_limit)

        path: Path = Path(path)
        self.path = path
        self._load_config()
    
    def _load_config(self):
        self.path.absolute().mkdir(parents=True, exist_ok=True)
        tables = self.path.glob("*/")

        config = {}
        for table in tables:
            config[table.name] = {
                "codes": [],
                "start": [],
                "end": [],
            }
            files = table.glob("[0-9]*-[0-9]*.parquet")
            codes = table / "codes.txt"
            with open(codes, 'r') as f:
                config[table.name]["codes"] = pd.Index(f.read().splitlines())

            for file in files:
                s, e = file.stem.split('-')
                config[table.name]["start"].append(s)
                config[table.name]["end"].append(e)
            config[table.name]["start"] = pd.to_datetime(config[table.name]["start"], errors='ignore').sort_values()
            config[table.name]["end"] = pd.to_datetime(config[table.name]["end"], errors='ignore').sort_values()
        self.config = config
    
    def __str__(self) -> str:
        output = super().__str__()
        for k, v in self.config.items():
            cs, ce = v['codes'][0], v['codes'][-1]
            ds, de = v['start'][0], v['end'][-1]
            output += f"\n\t{k}: {cs} - {ce} ({ds} - {de})"
        return output

    def __repr__(self) -> str:
        return self.__str__()

    def _write_col(self, table_path: Path, columns: list):
        with open(table_path / "codes.txt", "w") as f:
            for col in columns:
                f.write(col + "\n")
    
    def _write_table(self, table_path: Path, data: pd.DataFrame):
        size = data.memory_usage(deep=True).sum()
        item = data.shape[0]
        while size > self.size_limit or item > self.item_limit:
            size_idx = int((self.size_limit / data.memory_usage().sum()) * data.shape[0])
            item_idx = min(self.item_limit, data.shape[0])
            
            partition_idx = min(size_idx, item_idx)
            start = data.index[0].strftime('%Y%m%d')
            end = data.index[partition_idx].strftime('%Y%m%d')
            data.iloc[:partition_idx, :].to_parquet(table_path / f'{start}-{end}.parquet')
            data = data.iloc[partition_idx:, :]
            size = data.memory_usage(deep=True).sum()
        
        start = (data.index[0] if not isinstance(data.index[0], pd.Timestamp) 
                 else data.index[0].strftime('%Y%m%d'))
        end = (data.index[-1] if not isinstance(data.index[-1], pd.Timestamp) 
               else data.index[-1].strftime('%Y%m%d'))
        data.to_parquet(table_path / f'{start}-{end}.parquet')

    def _create(self, name: str, data: pd.DataFrame):
        data = data.sort_index()

        table_path = self.path / name
        table_path.mkdir()
        codes = data.columns

        self._write_col(table_path, codes)
        self._write_table(table_path, data)
    
    def _update(self, name: str, data: pd.DataFrame):
        data = data.sort_index()

        table_path = self.path / name
        codes = data.columns
        with open(table_path / "codes.txt", "r") as f:
            codes_old = f.readlines()
        codes_old = pd.Index(codes_old)
        if codes != codes_old:
            data_old = pd.read_parquet(table_path)
        data = pd.concat([data_old, data], axis=0, join='outer')

        self._write_col(table_path, codes)
        self._write_table(table_path, data)

    def dump(
        self, 
        data: pd.DataFrame, name: str = None
    ) -> 'Database':
        data = super().dump(data, name)
        
        for n, d in data.items():
            table_path = self.path / n
            if table_path.exists():
                self._update(n, d)
            else:
                self._create(n, d)
        self._load_config()
        return self
                
    def load(
        self,
        code: str | list,
        field: str | list,
        start: str | list = None,
        end: str = None,
        retdf: bool = False
    ) -> pd.DataFrame:
        field = qf.parse_commastr(field)
        code = qf.parse_commastr(code)

        result = {}
        for f in field:
            conf = self.config[f]
            start = qf.parse_date(start, default_date=conf["start"][0])
            end = qf.parse_date(end, default_date=conf["end"][-1])

            if not isinstance(start, list):
                start_max = conf["start"][conf["start"] <= start][-1]
                end_min = conf["end"][conf["end"] >= end][0]
                from_idx = conf["start"].get_loc(start_max)
                to_idx = conf["end"].get_loc(end_min)
                file = []
                for i in range(from_idx, to_idx + 1):
                    s, e = conf["start"][i], conf["end"][i]
                    s = s.strftime("%Y%m%d") if not isinstance(s, str) else s
                    e = e.strftime("%Y%m%d") if not isinstance(e, str) else e
                    file.append((self.path / f) / (s + '-' + e + '.parquet'))

                df = pd.read_parquet(file, columns=code)
                result[f] = df.loc[start:end]

            elif isinstance(start, list) and end is None:
                file = []
                for s in start:
                    end_min = conf["end"][conf["end"] >= s][0]
                    idx = conf["end"].get_loc(end_min)
                    s, e = conf["start"][idx], conf["end"][idx]
                    s = s.strftime("%Y%m%d") if not isinstance(s, str) else s
                    s = e.strftime("%Y%m%d") if not isinstance(e, str) else e
                    file.append((self.path / f) / (s + '-' + e + '.parquet'))

                df = pd.read_parquet(list(set(file)), columns=code)
                result[f] = df.loc[start]

            else:
                raise ValueError("Cannot assign start in a list type while end is not None")

        if not retdf:
            return result
        else:
            df = []
            for n, d in result.items():
                d = d.stack()
                d.name = n
                df.append(d)
            return pd.concat(df, axis=1)


if __name__ == "__main__":
    db = Database('/home/kali/data/other')
    df = ProxyFetcher().run()
    db.dump(df, 'proxy')