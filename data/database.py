import numpy as np
import pandas as pd
from pathlib import Path
from quantframe import (
    DatabaseBase,
    parse_commastr, 
    parse_date
)


class Database(DatabaseBase):

    def __init__(
        self, path: str = './database',
        size_limit: int = 100 * 2 ** 20,
        item_limit: int = 1e10,
        day_limit: int = 1e10,
    ) -> None:
        self.size_limit = int(size_limit)
        self.item_limit = int(item_limit)
        self.day_limit = int(day_limit)

        path: Path = Path(path)
        self.path = path
        path.absolute().mkdir(parents=True, exist_ok=True)
        tables = path.glob("*/")

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
            config[table.name]["start"] = pd.to_datetime(config[table.name]["start"]).sort_values()
            config[table.name]["end"] = pd.to_datetime(config[table.name]["end"]).sort_values()
        self.config = config
    
    def __str__(self) -> str:
        output = super().__str__()
        for k, v in self.config.items():
            cs, ce = v['codes'][0], v['codes'][-1]
            ds, de = v['start'][0].strftime('%Y-%m-%d'), v['end'][-1].strftime('%Y-%m-%d')
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
        day = data.index.date.size
        while size > self.size_limit or item > self.item_limit or day > self.day_limit:
            size_idx = int((self.size_limit / data.memory_usage().sum()) * data.shape[0])
            item_idx = min(self.item_limit, data.shape[0])
            day_idx = data.index.get_loc(data.loc[
                data.index.date[:self.day_limit][-1].strftime('%Y-%m-%d')].name)
            
            partition_idx = min(size_idx, item_idx, day_idx)
            start = data.index[0].strftime('%Y%m%d')
            end = data.index[partition_idx].strftime('%Y%m%d')
            data.iloc[:partition_idx, :].to_parquet(table_path / f'{start}-{end}.parquet')
            data = data.iloc[partition_idx:, :]
            size = data.memory_usage(deep=True).sum()
        
        start = data.index[0].strftime('%Y%m%d')
        end = data.index[-1].strftime('%Y%m%d')
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

    def dump(self, data: pd.DataFrame, name: str = None) -> None:
        data = super().dump(data, name)
        
        for n, d in data.items():
            table_path = self.path / n
            if table_path.exists():
                self._update(n, d)
            else:
                self._create(n, d)
                
    def load(
        self,
        code: str | list,
        field: str | list,
        start: str | list = None,
        end: str = None,
        retdf: bool = False
    ) -> pd.DataFrame:
        field = parse_commastr(field)
        code = parse_commastr(code)

        result = {}
        for f in field:
            conf = self.config[f]
            start = parse_date(start, default_date=conf["start"][0])
            end = parse_date(end, default_date=conf["end"][-1])

            if not isinstance(start, list):
                start_max = conf["start"][conf["start"] <= start][-1]
                end_min = conf["end"][conf["end"] >= end][0]
                from_idx = conf["start"].get_loc(start_max)
                to_idx = conf["end"].get_loc(end_min)
                file = []
                for i in range(from_idx, to_idx + 1):
                    file.append(
                        (self.path / f) / (
                            conf["start"][i].strftime("%Y%m%d") + '-' + 
                            conf["end"][i].strftime("%Y%m%d") + '.parquet'
                        )
                    )

                df = pd.read_parquet(file, columns=code)
                result[f] = df.loc[start:end]

            elif isinstance(start, list) and end is None:
                file = []
                for s in start:
                    end_min = conf["end"][conf["end"] >= s][0]
                    idx = conf["end"].get_loc(end_min)
                    file.append(
                        (self.path / f) / (
                            conf["start"][idx].strftime('%Y%m%d') + '-' + 
                            conf["end"][idx].strftime('%Y%m%d') + '.parquet'
                        ))

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
                    
