import genforge as qf
from genforge.database import DatabaseBase


class Asset:

    def __init__(
        self, 
        code: str, 
        table: Table = None
    ) -> None:
        self.table = table
        self.code = code
        self.universe_type = Universe
    
    def __add__(self, other: 'Universe | Asset') -> 'Universe':
        if isinstance(other, Universe):
            return other + self
        elif isinstance(other, Asset):
            return self.universe_type(self, other, table=self.table)
    
    def __str__(self):
        return f'Asset: {self.code}'
    
    def __repr__(self) -> str:
        return self.__str__()

    def load(
        self, 
        field: str | list = None,
        start: str = None,
        stop: str = None,
        date_index: str = "date",
        code_index: str = "order_book_id",
    ) -> pd.Series | pd.DataFrame:
        if not self.table:
            raise ValueError(f"{self} has no database")
        start = start or "2000-01-04"
        stop = stop or datetime.datetime.today().strftime(r'%Y-%m-%d')
        return self.table.load(
            field, [
                (date_index, ">=", parse_date(start)), 
                (date_index, "<=", parse_date(stop)), 
                (code_index, "=", self.code)
            ]
        )


class Universe:

    def __init__(
        self, 
        *asset: list[Asset] | list[str],
        table: Table = None
    ) -> None:
        self.asset = []
        self.code = []
        self.table = table
        for a in asset:
            if isinstance(a, Asset):
                self.code.append(a.code)
                self.asset.append(a)
            elif isinstance(a, str):
                self.code.append(a)
                self.asset.append(Asset(a, table=table))
    
    def __add__(self, other: 'Asset | Universe') -> 'Universe':
        if isinstance(other, Asset):
            self.asset.append(other)
            return self
        if isinstance(other, Universe):
            self.asset += other.asset
            return self
    
    def __str__(self):
        return f'Universe: {self.asset[:5]}'

    def __repr__(self) -> str:
        return self.__str__()
    
    def load(
        self, 
        field: str | list = None,
        start: str = None,
        stop: str = None,
        date_index: str = "date",
        code_index: str = "order_book_id",
    ):
        if not self.table:
            raise ValueError(f"{self} has no database")
        start = start or "2000-01-04"
        stop = stop or datetime.datetime.today().strftime(r'%Y-%m-%d')
        return self.table.load(
            field, [
                (date_index, ">=", parse_date(start)), 
                (date_index, "<=", parse_date(stop)), 
                (code_index, "in", self.code)
            ]
        )
