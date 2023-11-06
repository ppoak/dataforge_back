import datetime
import pandas as pd
import genforge as gf
from pathlib import Path


class Asset(gf.Table):

    def __init__(
        self,
        code: str | list[str],
        uri: str | Path,
        code_index: str = "order_book_id",
        date_index: str = "date",
    ):
        self.path = uri
        self.code = gf.parse_commastr(code)
        self.data = None
        self.code_index = code_index
        self.date_index = date_index
    
    def read(
        self, 
        field: str | list = None,
        start: str | list = None,
        stop: str = None,
    ) -> pd.Series | pd.DataFrame:
        field = gf.parse_date(field or "close")
        start = gf.parse_date(start or "2000-01-04")
        stop = gf.parse_date(stop or datetime.datetime.today().strftime(r'%Y-%m-%d'))

        if isinstance(start, list) and stop is not None:
            raise ValueError("If start is list, stop should be None")
        
        if (not isinstance(start, list) and self.data is not None and
            pd.Index(field).isin(self.data.columns).all() and
            self.data.index.min() < start and
            self.data.index.max() > stop):
            return self.data.loc[start:stop, field]
        
        elif (isinstance(start, list) and self.data is not None and
              pd.Index(field).isin(self.data.columns).all() and
              self.data.index.get_level_values(self.date_index).isin(start).all()):
            return self.data.loc[start, field]
        
        elif not isinstance(start, list):
            self.data = self.read(
                field, [
                    (self.date_index, ">=", gf.parse_date(start)), 
                    (self.date_index, "<=", gf.parse_date(stop)), 
                    (self.code_index, "=", self.code)
                ]
            )
            return self.data
        
        elif isinstance(start, list) and stop is None:
            self.data = self.read(
                field, [
                    (self.date_index, "in", gf.parse_date(start)),
                    (self.code_index, "=", self.code),
                ]
            )
            return self.data
        
        else:
            raise ValueError("Invalid start, stop or field values")




