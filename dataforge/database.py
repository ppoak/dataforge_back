import datetime
import pandas as pd
from pathlib import Path
from typing import Callable
from .tools import parse_commastr, parse_date


class Table:

    def __init__(
        self,
        uri: str | Path,
        spliter: str | list | dict | pd.Series | Callable | None = None,
        namer: str | list | dict | pd.Series | Callable | None = None,
    ):
        self.path = Path(uri).expanduser().resolve()
        self.name = self.path.stem
        if not ((spliter is None and namer is None) or 
                (spliter is not None and namer is not None)):
            raise ValueError('spliter and namer must be both None or both not None')
        self.spliter = spliter
        self.namer = namer
    
    @property
    def fragments(self):
        return [f.stem for f in list(self.path.glob('**/*.parquet'))]

    def create(self):
        self.path.mkdir(parents=True, exist_ok=True)
    
    def read(
        self,
        columns: str | list[str] | None = None,
        filters: list[list[tuple]] = None,
    ):
        df = pd.read_parquet(
            self.path, 
            engine = 'pyarrow', 
            columns = parse_commastr(columns),
            filters = filters,
        )
        return df
    
    def _read_fragment(
        self,
        fragment: list | str = None,
    ):
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        fragment = [self.path.joinpath(frag).with_suffix('.parquet') for frag in fragment]
        return pd.read_parquet(fragment, engine='pyarrow')
    
    def update(
        self,
        df: pd.DataFrame,
        fragment: str = None,
    ):
        fragment = fragment or self.name
        df = pd.concat([self._read_fragment(fragment), df], axis=0)
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        if self.spliter:
            df.groupby(self.spliter).apply(
                lambda x: x.to_parquet(
                    f"{self.path.joinpath(self.namer(x))}.parquet"
            ))
        else:
            df.to_parquet(self.path.joinpath(fragment).with_suffix('.parquet'))
    
    def write(
        self,
        df: pd.DataFrame,
        fragment: str = None,
    ):
        fragment = fragment or self.name
        if not isinstance(fragment, str):
            raise ValueError("fragment should be in string format")
        if self.spliter:
            df.loc[~df.index.duplicated(keep='last')].sort_index()\
            .groupby(self.spliter).apply(
                lambda x: x.to_parquet(
                    f"{self.path.joinpath(self.namer(x))}.parquet"
            ))
        else:
            df.to_parquet(self.path.joinpath(fragment).with_suffix('.parquet'))

    def remove(
        self,
        fragment: str | list = None,
    ):
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        for frag in fragment:
            (self.path.joinpath(frag).with_suffix('.parquet')).unlink()
    
    def __str__(self) -> str:
        return f'Table at <{self.path.absolute()}>'
    
    def __repr__(self) -> str:
        return self.__str__()


class AssetTable(Table):

    def __init__(
        self,
        uri: str | Path,
        date_index: str = 'date',
        code_index: str = 'order_book_id',
    ):
        spliter = lambda x: x[1].year * 100 + x[1].month
        namer = lambda x: x.index.get_level_values(1)[0].strftime(r'%Y%m')
        super().__init__(uri, spliter, namer)
        self.date_index = date_index
        self.code_index = code_index
    
    def read(
        self, 
        field: str | list = None,
        code: str | list = None,
        start: str | list = None,
        stop: str = None,
    ) -> pd.Series | pd.DataFrame:
        code = parse_commastr(code)
        field = parse_date(field or "close")
        start = parse_date(start or "20000104")
        stop = parse_date(stop or datetime.datetime.today().strftime(r'%Y%m%d'))

        if isinstance(start, list) and stop is not None:
            raise ValueError("If start is list, stop should be None")
                
        elif not isinstance(start, list):
            filters = [
                (self.date_index, ">=", parse_date(start)), 
                (self.date_index, "<=", parse_date(stop)), 
            ]
            if code is not None:
                filters.append((self.code_index, "in", code))
            return super().read(field, filters)
        
        elif isinstance(start, list) and stop is None:
            filters = [(self.date_index, "in", parse_date(start))]
            if code is not None:
                filters.append((self.code_index, "in", code))
            return super().read(field, filters)
        
        else:
            raise ValueError("Invalid start, stop or field values")
    
    def update(
        self, df: pd.DataFrame, 
    ):
        fragment = sorted(self.fragments)[-1]
        super().update(df, fragment)
        

class FrameTable(Table):

    def read(
        self,
        column: str | list = None,
        index: str | list = None,
        index_name: str = 'order_book_id',
    ):
        filters = None
        if index is not None:
            filters = [(index_name, "in", parse_commastr(index))]
        super().read(parse_commastr(column), filters)

