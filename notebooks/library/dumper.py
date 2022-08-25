import re
import abc
import numpy as np
import pandas as pd
from pathlib import Path


class FileDumperBase(abc.ABC):
    SEP = '\t'
    SUFFIX = 'bin'
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
        file_path: str,
        file_type: str = "csv",
        date_field: str = None,
        inst_field: str = None,
        dump_field: 'list | str' = None,
        dump_path: str = "./qlib-data",
        dump_mode: str = "w",
        name_pattern: str = None,
        name_col: str = None,
        freq: str = "day",
    ):
        """Base Class for dump file base data to qlib format"""

        assert \
            file_type in ['csv', 'parquet', 'feather', 'pickle'], \
            f"Currently {file_type} is not supported"
        
        assert \
            not (name_pattern is not None and name_col is None), \
            "Cannot parse a None name_col with not None name_pattern"

        self.file_path = file_path
        self.file_type = file_type
        self.inst_field = inst_field
        self.date_field = date_field
        self.dump_field = dump_field
        self.dump_mode = dump_mode

        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)
        self.instrument_path = Path(self.dump_path).joinpath(FileDumperBase.INSTPATH)
        self.calendar_path = Path(self.dump_path).joinpath(FileDumperBase.CALPATH)
        self.feature_path = Path(self.dump_path).joinpath(FileDumperBase.FEATPATH)
        self.instrument_path.mkdir(parents=True, exist_ok=True)
        self.calendar_path.mkdir(parents=True, exist_ok=True)
        self.feature_path.mkdir(parents=True, exist_ok=True)
        
        self.name_pattern = name_pattern
        self.name_col = name_col
        self.freq = freq
    
    def _load_data(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement _load_data method")
    
    def _process_data(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement _process_data method")

    def _dump_inst(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")

    def _dump_cal(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")
    
    def _dump_feat(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")
        
    @abc.abstractclassmethod
    def dump(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")


class FileDumper(FileDumperBase):
    
    def _load_data(self, **kwargs):
        # in case set columns accidentally
        kwargs.update({"columns": self.dump_field})
        path = Path(self.file_path)
        data_reader = getattr(pd, f'read_{self.file_type}')
        if path.is_dir():
            datas = []
            for fp in path.glob(f'*.{self.file_type}'):
                data = data_reader(fp, **kwargs)
                if self.name_pattern is not None:
                    name = re.findall(self.name_pattern, fp.stem)[0]
                    data[self.name_col] = name
                datas.append(data)
            data = pd.concat(datas, axis=0)
        else:
            data = data_reader(path, **kwargs)

        self.data = data

        return self

    def _process_data(self):
        if self.date_field is None and self.inst_field is None:
            # this is the user's duty to ensure the index is multi-index
            # with date field and indstrument field
            # we make sure the converted data has the level0 datetime index format
            # and level1 object or category index format
            self.data.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(self.data.index.get_level_values(0), errors='ignore'), 
                pd.to_datetime(self.data.index.get_level_values(1), errors='ignore')],
            )
            if not isinstance(self.data.index.levels[0], pd.DatetimeIndex):
                self.data = self.data.swaplevels()
        
        elif self.date_field is not None and self.inst_field is None:
            # this means that the intruments is index while date fields isn't
            self.data.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(self.data[self.date_field]), self.data.index]
            )
            self.data.drop([self.date_field], axis=1, inplace=True)
        
        elif self.date_field is None and self.inst_field is not None:
            self.data.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(self.data.index), self.data[self.inst_field]]
            )
            self.data.drop([self.inst_field], axis=1, inplace=True)
        
        else:
            self.data.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(self.data[self.date_field]), self.data[self.inst_field]]
            )
            self.data.drop([self.inst_field, self.date_field], axis=1, inplace=True)

        return self
    
    def _dump_cal(self):
        self.calendar = pd.DatetimeIndex(
            self.calendar if self.dump_mode != "a" else 
            sorted(list(set(self.calendar[self.calendar > self.old_calendar.min()]) | set(self.old_calendar)))
        )
        pd.Series(
            self.calendar,
        ).map(lambda x: x.strftime(FileDumperBase.TIMEFMT if self.freq == "day" else FileDumperBase.HFREQTIMEFMT)
        ).to_csv(
            self.calendar_path.joinpath(f'{self.freq}.txt'), 
            index=False, 
            header=False
        )
        return self

    def _dump_inst(self):
        start_date = self.data.groupby(level=1).apply(lambda x: x.index.get_level_values(0).min())
        end_date = self.data.groupby(level=1).apply(lambda x: x.index.get_level_values(0).max())
        self.instruments = pd.concat([start_date, end_date], axis=1)

        if self.dump_mode == "a":
            self.instruments.columns = [FileDumperBase.INSTSTARTNAME, FileDumperBase.INSTENDNAME]
            common_idx = self.instruments.index.intersection(self.old_instruments.index)
            new_idx = self.instruments.index.difference(self.old_instruments.index)
            self.instruments = pd.concat([self.old_instruments, self.instruments.loc[new_idx]], axis=0)
            self.instruments[FileDumperBase.INSTSTARTNAME].mask(
                self.instruments.loc[common_idx, FileDumperBase.INSTSTARTNAME] <
                self.old_instruments.loc[common_idx, FileDumperBase.INSTSTARTNAME],
                self.old_instruments[FileDumperBase.INSTSTARTNAME], inplace=True,
            )
            self.instruments[FileDumperBase.INSTENDNAME].mask(
                self.instruments.loc[common_idx, FileDumperBase.INSTENDNAME] < 
                self.old_instruments.loc[common_idx, FileDumperBase.INSTENDNAME],
                self.old_instruments[FileDumperBase.INSTENDNAME], inplace=True,
            )

        self.instruments.to_csv(str(self.instrument_path.joinpath(
            f'{FileDumperBase.ALLINSTNAME}.txt').resolve()), header=False, sep=FileDumperBase.SEP, index=True)
        
        return self
        
    def _dump_feat(self):
        if self.dump_mode == "a":
            update_calendar = sorted(list(set(self.calendar) - set(self.old_calendar)))
            update_cdata = self.data.loc(axis=0)[update_calendar, self.old_instruments.index]
            update_instruments = sorted(list(set(self.instruments.index) - set(self.old_instruments.index)))
            update_idata = self.data.loc(axis=0)[self.calendar, update_instruments]
        else:
            update_idata = self.data
            update_cdata = pd.DataFrame()

        def _ensure_path(code):
            code_path = self.feature_path.joinpath(code.lower())
            code_path.mkdir(parents=True, exist_ok=True)
            return code_path

        def _overwrite(data):
            code_path = _ensure_path(data.index.get_level_values(1)[0])
            for feat in self.data.columns:
                np.hstack([self.calendar.get_indexer(data.index.get_level_values(0)[:1])[0], 
                    data[feat].values]).astype('float32').tofile(
                        code_path.joinpath(f'{feat}.{self.freq}.{FileDumperBase.SUFFIX}'))
        
        def _update(data):
            code_path = _ensure_path(data.index.get_level_values(1)[0])
            for feat in self.data.columns:
                with open(code_path.joinpath(f'{feat}.{self.freq}.{FileDumperBase.SUFFIX}'), 'ab') as f: 
                    data.loc[:, feat].values.astype('float32').tofile(f)

        if not update_cdata.empty:
            # only 'a' mode can access this
            update_cdata.groupby(level=1).apply(_update)
        if not update_idata.empty:
            # no matter 'a' mode or 'w' mode, new instrument must be created
            update_idata.groupby(level=1).apply(_overwrite)
        
        print('Your data is now up-to-date!')
        return self
        
    def dump(self, **kwargs):
        if self.dump_mode == "a":
            self.old_calendar = pd.read_csv(self.calendar_path.joinpath(f"{self.freq}.txt"), 
                header=None, parse_dates=[0]).iloc[:, 0]
            self.old_instruments = pd.read_csv(self.instrument_path.joinpath("all.txt"), 
                header=None, sep='\t', index_col=0, parse_dates=[1, 2])
            self.old_instruments.columns = [FileDumperBase.INSTSTARTNAME, FileDumperBase.INSTENDNAME]
        
        self._load_data(**kwargs)._process_data()
        self.calendar = self.data.index.levels[0].sort_values()

        self._dump_cal()._dump_inst()._dump_feat()


class IndexCompDumper(FileDumper):

    def __init__(
        self,
        file_path: str,
        file_type: str = "csv",
        date_field: str = None,
        inst_field: str = None,
        index_field: str = 0,
        dump_path: str = "./qlib-data",
        name_pattern: str = None,
        name_col: str = None,
    ):
        assert index_field is not None, "Index field cannot be None, use index level like 0 instead"
        super().__init__(
            file_path,
            file_type,
            date_field,
            inst_field,
            None,
            dump_path,
            'w',
            name_pattern,
            name_col,
            'day'
        )
        self.index_field = index_field

    def _process_data(self):
        if isinstance(self.index_field, int):
            self.index_field = self.data.index.names[self.index_field] or f'level_{self.index_field}'
            self.data.reset_index(self.index_field, inplace=True)
        
        super()._process_data()        
        return self

    def _dump_inst(self):
        def _save(data):
            entry_date = data.groupby(level=1).apply(lambda x: x.index.get_level_values(0).min())
            exit_date = data.groupby(level=1).apply(lambda x: x.index.get_level_values(0).max())
            instruments = pd.concat([entry_date, exit_date], axis=1)
            instruments.applymap(lambda x: x.strftime(FileDumperBase.TIMEFMT)
                ).to_csv(
                    str(self.instrument_path.joinpath(
                        data[self.index_field].iloc[0].lower() + '.txt'
                    ).resolve()),
                    sep = FileDumperBase.SEP, header = False,
                )
        self.data.groupby(self.index_field).apply(_save)
        return self

    def _dump_cal(self):
        return self

    def _dump_feat(self):
        return self

    def dump(self):
        self._load_data()._process_data()._dump_inst()
    

if __name__ == "__main__":
    dumper = FileDumper(
        file_path = "data/kline-day/daily_stock_trade_data.feather",
        file_type = "feather",
        date_field = "date",
        inst_field = "stock_code",
        dump_field = None,
        dump_path = "./data/qlib-day",
        dump_mode = "w",
        name_pattern = None,
        freq = "day",
    )
    dumper.dump()

    dumper = FileDumper(
        file_path = "data/index-market-daily/index-market-daily.parquet",
        file_type = "parquet",
        date_field = None,
        inst_field = None,
        dump_field = None,
        dump_path = "./data/qlib-day",
        dump_mode = "a",
        name_pattern = None,
        freq = "day",
    )
    dumper.dump()

    dumper = IndexCompDumper(
        file_path = "./data/index-weights/index-weights.parquet",
        file_type = "parquet",
        date_field = None,
        inst_field = None,
        index_field = 1,
        dump_path = "./data/qlib-day/",
        name_pattern = None,
        name_col = None
    )
    dumper.dump()
