import re
import numpy as np
import pandas as pd
from pathlib import Path


class FileDumper:
    SEP = '\t'
    SUFFIX = 'bin'
    IDXPRESENT = 'index'
    ALLINSTNAME = 'all'
    INSTPATH = 'instruments'
    CALPATH = 'calendars'
    FEATPATH = 'features'
    TIMEFMT = "%Y-%m-%d"
    HFREQTIMEFMT = "%Y-%m-%d %H:%M:%S"
    INSTSTARTNAME = "entry_date"
    INSTENDNAME = "exit_date"

    def __init__(
        self,
        file_path: 'str | list',
        file_type: str,
        date_field: str,
        inst_field: str,
        dump_field: 'list | str',
        dump_path: str,
        dump_mode: str,
        name_pattern: str,
        freq: str,
    ):
        data = self._load_data(
            file_path, 
            file_type, 
            dump_field,
            inst_field,
            date_field,
            name_pattern,
        )

        if inst_field == FileDumper.IDXPRESENT and date_field == FileDumper.IDXPRESENT:
            if not isinstance(data.index.levels[1], pd.DatetimeIndex):
                data = data.swaplevel()
        elif inst_field == FileDumper.IDXPRESENT and date_field != FileDumper.IDXPRESENT:
            data.index = pd.MultiIndex.from_arrays([data.index, data[date_field]])
        elif inst_field != FileDumper.IDXPRESENT and date_field == FileDumper.IDXPRESENT:
            data.index = pd.MultiIndex.from_arrays([data[inst_field], data.index])
        else:
            data = data.set_index([inst_field, date_field])
        self.data = data

        self.freq = freq
        self.dump_mode = dump_mode
        self.instruments = data.index.levels[0].sort_values().to_list()
        self.calendar = data.index.levels[1].sort_values().to_list()

        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)
        self.instrument_path = Path(self.dump_path).joinpath(FileDumper.INSTPATH)
        self.calendar_path = Path(self.dump_path).joinpath(FileDumper.CALPATH)
        self.feature_path = Path(self.dump_path).joinpath(FileDumper.FEATPATH)
        self.instrument_path.mkdir(parents=True, exist_ok=True)
        self.calendar_path.mkdir(parents=True, exist_ok=True)
        self.feature_path. mkdir(parents=True, exist_ok=True)

        if self.dump_mode == "update":
            self.old_calendar = pd.read_csv(self.calendar_path.joinpath(f"{freq}.txt"), header=None, parse_dates=[0]).iloc[:, 0].to_list()
            self.old_instruments = pd.read_csv(self.instrument_path.joinpath("all.txt"), header=None, sep='\t', index_col=0)
            self.old_instruments.columns = [FileDumper.INSTSTARTNAME, FileDumper.INSTENDNAME]
    
    def _load_data(
        self, 
        file_path: str, 
        file_type: str, 
        dump_field: 'str | list', 
        inst_field: str,
        date_field: str,
        name_pattern: str,
    ):
        data_reader = eval(f'pd.read_{file_type}')
        if isinstance(file_path, str):
            data = data_reader(file_path, columns=dump_field)
        elif isinstance(file_path, (list, tuple)):
            datas = []
            for fp in file_path:
                data = data_reader(fp, columns=dump_field)
                if name_pattern is not None:
                    name = re.findall(name_pattern, Path(file_path).stem)[0]
                    data[inst_field or date_field] = name if inst_field else pd.to_datetime(name)
                datas.append(data)
            data = pd.concat(datas, axis=0)
        return data
    
    def dump_inst(self):
        start_date = self.data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).min())
        end_date = self.data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).max())
        instruments = pd.concat([start_date, end_date], axis=1)
        instruments.columns = [FileDumper.INSTSTARTNAME, FileDumper.INSTENDNAME]

        if self.dump_mode == "update":
            instruments[FileDumper.INSTSTARTNAME].mask(
                self.old_instruments[FileDumper.INSTSTARTNAME] < instruments[FileDumper.INSTSTARTNAME],
                self.old_instruments[FileDumper.INSTSTARTNAME], inplace=True,
            )
            instruments[FileDumper.INSTENDNAME].mask(
                self.old_instruments[FileDumper.INSTENDNAME] > instruments[FileDumper.INSTENDNAME],
                self.old_instruments[FileDumper.INSTENDNAME], inplace=True,
            )

        instruments.to_csv(self.instrument_path. joinpath(f'{FileDumper.ALLINSTNAME}.txt'), header=False, sep=FileDumper.SEP, index=True)
    
    def dump_cal(self):
        self.calendar = (
            self.calendar if self.dump_mode != "update" else 
            sorted(list(set(self.calendar) | set(self.old_calendar)))
        )
        pd.Series(
            self.calendar,
        ).map(lambda x: x.strftime(FileDumper.TIMEFMT if self.freq == "day" else FileDumper.HFREQTIMEFMT)
        ).to_csv(
            self.calendar_path.joinpath(f'{self.freq}.txt'), 
            index=False, 
            header=False
        )
    
    def dump_feat(self):

        def _ensure_path(code):
            code_path = self.feature_path.joinpath(code.lower())
            code_path.mkdir(parents=True, exist_ok=True)
            return code_path

        def _overwrite(data):
            code_path = _ensure_path(data.index.get_level_values(0)[0])
            for feat in self.data.columns:
                np.hstack([self.calendar.index(data.index.get_level_values(1)[0]), 
                    data[feat].values]).astype('float32').tofile(
                        code_path.joinpath(f'{feat}.{FileDumper.FREQ}.{FileDumper.SUFFIX}'))
        
        def _update(data):
            code_path = _ensure_path(data.index.get_level_values(0)[0])
            for feat in self.data.columns:
                with open(code_path.joinpath(f'{feat}.{FileDumper.FREQ}.{FileDumper.SUFFIX}'), 'ab') as f: 
                        data[feat].values.astype('float32').tofile(f)
        
        self.data.groupby(level=0).apply(_overwrite if self.dump_mode != "update" else _update)
        
    def dump(self):
        self.dump_inst()
        self.dump_cal()
        self.dump_feat()


if __name__ == "__main__":
    dumper = FileDumper(
        file_path = "./data/kline-daily/market-daily.parquet",
        file_type = "parquet",
        date_field = "index",
        inst_field = "index",
        dump_field = None,
        dump_path = "./data/qlib-day",
        dump_mode = "update",
        freq = "day"
    )
    dumper.dump_cal()
