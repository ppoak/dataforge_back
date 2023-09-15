import re
import time
import random
import datetime
import requests
import numpy as np
import pandas as pd
import akshare as ak
from tqdm import tqdm
import quantframe as qf
from math import ceil
from lxml import etree
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import quote
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

def strip_stock_code(code: str):
    code_pattern = r'\.?[Ss][Zz]\.?|\.?[Ss][Hh]\.?|\.?[Bb][Jj]\.?'\
        '|\.?[Oo][Ff]\.?'
    return re.sub(code_pattern, '', code)

def reduce_mem_usage(df: pd.DataFrame):
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
            ds = ds if not isinstance(ds, pd.Timestamp) else ds.strftime(r'%Y-%m-%d')
            de = de if not isinstance(de, pd.Timestamp) else de.strftime(r'%Y-%m-%d')
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
    ashare_db = Database('/home/kali/data/ashare')
    print(ashare_db.load(
        '600000.XSHG, 600519.XSHG, 000001.XSHE',
        'close, adjfactor',
        '20200101',
        '20201231'
    ))
