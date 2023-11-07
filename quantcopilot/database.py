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
    ):
        self.path = Path(uri).expanduser().resolve()
        self.name = self.path.stem
        self.spliter = spliter
    
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
        return pd.read_parquet(fragment, engine='pyarrow')
    
    def write(
        self,
        df: pd.DataFrame,
        fragment: str = None,
    ):
        fragment = fragment or self.name
        df = pd.concat([self._read_fragment(fragment), df], axis=0)
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        if self.spliter:
            spilter = self.spliter
            df.groupby(spilter, as_index=True).apply(
                lambda x: x.droplevel(0).to_parquet(
                    f"{self.path.joinpath(x.index.get_level_values(0)[0])}.parquet"
            ))
        else:
            df.to_paruqet(self.path.joinpath(fragment).with_suffix('.parquet'))
    
    def update(
        self,
        value: pd.Series | list | str,
        index: pd.Index | list | str = None,
        column: pd.Index | list | str = None,
        fragment: list | str = None,
    ):
        fragment = fragment or self.fragments
        df = self._read_fragment(fragment)
        index = index or df.index
        column = column or df.columns
        df.loc[index, column] = value
        if self.spliter:
            spilter = self.spliter
            df.groupby(spilter, as_index=True).apply(
                lambda x: x.droplevel(0).to_parquet(
                    f"{self.path.joinpath(x.index.get_level_values(0)[0])}.parquet"
            ))
        else:
            df.to_parquet(self.path.joinpath(fragment).with_suffix('.parquet'))
    
    def drop(
        self,
        index: pd.Index | list | str = None,
        column: pd.Index | list | str = None,
        level: int | list | str = None,
        fragment: list | str = None,
    ):
        fragment = fragment or self.fragments
        df = self._read_fragment(fragment)
        df = df.drop(index=index, columns=column, level=level)
        if self.spliter:
            spilter = self.spliter
            df.groupby(spilter, as_index=True).apply(
                lambda x: x.droplevel(0).to_parquet(
                    f"{self.path.joinpath(x.index.get_level_values(0)[0])}.parquet"
            ))
        else:
            df.to_parquet(self.path.joinpath(fragment).with_suffix('.parquet'))

    def remove(
        self,
        fragment: str | list = None,
    ):
        fragment = fragment or self.fragments
        for frag in fragment:
            frag.unlink()
    
    def __str__(self) -> str:
        return f'Table at <{self.path.absolute()}>'
    
    def __repr__(self) -> str:
        return self.__str__()


class AssetTable(Table):

    def __init__(
        self,
        uri: str | Path,
        code_index: str = "order_book_id",
        date_index: str = "date",
    ):
        spliter = lambda x: x.year * 100 + x.month
        super().__init__(uri, spliter)
        self.code_index = code_index
        self.date_index = date_index
    
    def read(
        self, 
        code: str | list = None,
        field: str | list = None,
        start: str | list = None,
        stop: str = None,
    ) -> pd.Series | pd.DataFrame:
        code = parse_commastr(code)
        field = parse_date(field or "close")
        start = parse_date(start or "2000-01-04")
        stop = parse_date(stop or datetime.datetime.today().strftime(r'%Y-%m-%d'))

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
    
    def write(
        self, df: pd.DataFrame, 
    ):
        fragment = sorted(self.fragments)[-1]
        super().write(df, fragment)
        

class Proxy(Table):

    pass
