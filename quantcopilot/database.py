import re
import numpy as np
import pandas as pd
import genforge as qf
from pathlib import Path


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


class PanelTable(qf.Table):

    def load(
        self, 
        field: str | list[str] | None = None,
        start: str | pd.Timestamp | list[str] = None,
        stop: str | pd.Timestamp = None,
        code: str | list[str] = None,
        date_name: str = 'date',
        code_name: str = 'code',
    ):
        field = qf.parse_commastr(field)
        filters = []
        start = qf.parse_date(start)
        stop = qf.parse_date(stop)
        code = qf.parse_commastr(code)
        if isinstance(start, list) and stop is not None:
            raise ValueError("When assigning a list of time, `stop` should be None")
        elif isinstance(start, list):
            filters.append([date_name, "in", start])
        elif isinstance(start, pd.Timestamp):
            filters.append([date_name, ">=", start])
        if isinstance(stop, pd.Timestamp):
            filters.append([date_name, "<=", stop])
        if code is not None:
            filters.append([code_name, "in", code])
        return super().load(field, filters)


class FrameTable(qf.Table):

    def load(
        self, 
        field: str | list[str] = None,
        index: str | list[str] | pd.Timestamp = None,
        name: str = "__index_level0__"
    ):
        field = qf.parse_commastr(field)
        index = qf.parse_commastr(index)
        filters = []
        if isinstance(index, list):
            filters.append([name, "in", index])
        elif index is not None:
            filters.append([name, "=", index])
        return super().load(field, filters)
