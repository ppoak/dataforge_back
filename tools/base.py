import re
import numpy as np


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
