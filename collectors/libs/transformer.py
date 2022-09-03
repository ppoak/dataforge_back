import pandas as pd
from pathlib import Path


class Transformer:

    def __init__(
        self,
        data,
        inst_col: 'int | str' = None,
        date_col: 'int | str' = None,
        uri: str = './data',
        by: str = 'inst',
    ):
        self._data = data.copy()
        
        if inst_col is None and date_col is None:
            if not isinstance(data.index, pd.MultiIndex):
                raise ValueError('If both date_col and inst_col are None, '
                    'Your data must be MultiIndexed')
            try:
                pd.to_datetime(data.index.levels[0], errors='raise')
                date_col = 0
                inst_col = 1
            except:
                date_col = 1
                inst_col = 0
        else:
            inst_col = inst_col or 0
            date_col = date_col or 0
        self._inst_col = inst_col
        self._date_col = date_col
        
        self._uri = Path(uri)
        self._uri.mkdir(parents=True, exist_ok=True)
        self._by = by
    
    def _process_data(self):
        if not isinstance(self._inst_col, int) and not isinstance(self._date_col, int):
            self._data[self._date_col] = pd.to_datetime(self._data[self._date_col])
            self._data = self._data.set_index([self._date_col, self._inst_col])
            self._date_col = 0
            self._inst_col = 1
        elif not isinstance(self._inst_col, int) and isinstance(self._date_col, int):
            self._data = self._data.set_index(self._inst_col, append=True)
            self._date_col = 0
            self._inst_col = 1
        elif isinstance(self._inst_col, int) and not isinstance(self._date_col, int):
            self._data.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(self._data[self._date_col]), self._data.index]
            )
            self._date_col = 0
            self._inst_col = 1
        
        return self
    
    def transform(self):
        def _save(data):
            code = data.index.get_level_values(self._inst_col)[0]
            date_range = (
                f"{data.index.get_level_values(self._date_col).min().strftime('%Y%m%d')}-"
                f"{data.index.get_level_values(self._date_col).max().strftime('%Y%m%d')}"
            )
            code_path = self._uri.joinpath(f'{code}')
            code_path.mkdir(parents=True, exist_ok=True)
            data.droplevel(self._inst_col).to_parquet(
                code_path.joinpath(f'{date_range}.parquet'), compression='gzip')
        self._data.groupby(level=self._inst_col).apply(_save)
