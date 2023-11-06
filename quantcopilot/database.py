import datetime
import pandas as pd
import dataforge as forge
from pathlib import Path


class QuotesDay(forge.Table):

    def __init__(
        self,
        uri: str | Path,
        code: str | list[str],
        code_index: str = "order_book_id",
        date_index: str = "date",
    ):
        super().__init__(uri)
        self.code = forge.parse_commastr(code)
        self.data = None
        self.code_index = code_index
        self.date_index = date_index
    
    def read(
        self, 
        field: str | list = None,
        start: str | list = None,
        stop: str = None,
    ) -> pd.Series | pd.DataFrame:
        field = forge.parse_date(field or "close")
        start = forge.parse_date(start or "2000-01-04")
        stop = forge.parse_date(stop or datetime.datetime.today().strftime(r'%Y-%m-%d'))

        if isinstance(start, list) and stop is not None:
            raise ValueError("If start is list, stop should be None")
        
        if (not isinstance(start, list) and self.data is not None and
            pd.Index(field).isin(self.data.columns).all() and
            pd.Index(self.code).isin(self.data.index.get_level_values(self.code_index)).all() and
            self.data.index.get_level_values(self.date_index).min() < start and
            self.data.index.get_level_values(self.date_index).max() > stop):
            return self.data.loc[(slice(start, stop), self.code), field]
        
        elif (isinstance(start, list) and self.data is not None and
              pd.Index(field).isin(self.data.columns).all() and
              pd.Index(self.code).isin(self.data.index.get_level_values(self.code_index)).all() and
              self.data.index.get_level_values(self.date_index).isin(start).all()):
            return self.data.loc[(start, self.code), field]
        
        elif not isinstance(start, list):
            self.data = self.read(
                field, [
                    (self.date_index, ">=", forge.parse_date(start)), 
                    (self.date_index, "<=", forge.parse_date(stop)), 
                    (self.code_index, "in", self.code)
                ]
            )
            return self.data
        
        elif isinstance(start, list) and stop is None:
            self.data = self.read(
                field, [
                    (self.date_index, "in", forge.parse_date(start)),
                    (self.code_index, "in", self.code),
                ]
            )
            return self.data
        
        else:
            raise ValueError("Invalid start, stop or field values")


class Proxy(forge.Table):

    pass
