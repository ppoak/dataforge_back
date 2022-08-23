import re
import abc
import numpy as np
import pandas as pd
from pathlib import Path


class FileDumperBase(abc.ABC):
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
        file_path: str,
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

        if inst_field == FileDumperBase.IDXPRESENT and date_field == FileDumperBase.IDXPRESENT:
            if not isinstance(data.index.levels[1], pd.DatetimeIndex):
                data = data.swaplevel()
        elif inst_field == FileDumperBase.IDXPRESENT and date_field != FileDumperBase.IDXPRESENT:
            data.index = pd.MultiIndex.from_arrays([data.index, data[date_field]])
        elif inst_field != FileDumperBase.IDXPRESENT and date_field == FileDumperBase.IDXPRESENT:
            data.index = pd.MultiIndex.from_arrays([data[inst_field], data.index])
        else:
            data = data.set_index([inst_field, date_field])
        self.data = data

        self.freq = freq
        self.dump_mode = dump_mode
        self.inst_field = inst_field
        self.date_field = date_field
        self.instruments = data.index.levels[0].sort_values().to_list()
        self.calendar = data.index.levels[1].sort_values().to_list()

        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)
        self.instrument_path = Path(self.dump_path).joinpath(FileDumperBase.INSTPATH)
        self.calendar_path = Path(self.dump_path).joinpath(FileDumperBase.CALPATH)
        self.feature_path = Path(self.dump_path).joinpath(FileDumperBase.FEATPATH)
        self.instrument_path.mkdir(parents=True, exist_ok=True)
        self.calendar_path.mkdir(parents=True, exist_ok=True)
        self.feature_path.mkdir(parents=True, exist_ok=True)

        if self.dump_mode == "update":
            self.old_calendar = pd.read_csv(self.calendar_path.joinpath(f"{freq}.txt"), 
                header=None, parse_dates=[0]).iloc[:, 0].to_list()
            self.old_instruments = pd.read_csv(self.instrument_path.joinpath("all.txt"), 
                header=None, sep='\t', index_col=0, parse_dates=[1, 2])
            self.old_instruments.columns = [FileDumperBase.INSTSTARTNAME, FileDumperBase.INSTENDNAME]
    
    def _load_data(
        self, 
        file_path: str, 
        file_type: str, 
        dump_field: 'str | list', 
        inst_field: str,
        date_field: str,
        name_pattern: str,
    ):
        file_path = Path(file_path)
        data_reader = eval(f'pd.read_{file_type}')
        if file_path.is_file():
            data = data_reader(file_path, columns=dump_field)
        elif file_path.is_dir():
            datas = []
            for fp in file_path.iterdir():
                data = data_reader(fp, columns=dump_field)
                if name_pattern is not None:
                    name = re.findall(name_pattern, fp.stem)[0]
                    data[inst_field or date_field] = name if inst_field else pd.to_datetime(name)
                datas.append(data)
            data = pd.concat(datas, axis=0)
        return data
    
    @abc.abstractclassmethod
    def dump_inst(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")

    @abc.abstractclassmethod
    def dump_cal(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")
    
    @abc.abstractclassmethod
    def dump_feat(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")
        
    @abc.abstractclassmethod
    def dump(self):
        raise NotImplementedError("Subclass of FileDumperBase must implement dump_* method")


class FileDumper(FileDumperBase):

    def dump_inst(self):
        start_date = self.data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).min())
        end_date = self.data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).max())
        instruments = pd.concat([start_date, end_date], axis=1)

        if self.dump_mode == "update":
            instruments.columns = [FileDumperBase.INSTSTARTNAME, FileDumperBase.INSTENDNAME]
            common_idx = instruments.index.intersection(self.old_instruments.index)
            new_idx = instruments.index.difference(self.old_instruments.index)
            instruments = pd.concat([self.old_instruments, instruments.loc[new_idx]], axis=0)
            instruments[FileDumperBase.INSTENDNAME].mask(
                instruments.loc[common_idx, FileDumperBase.INSTENDNAME] < 
                instruments.loc[common_idx, FileDumperBase.INSTENDNAME],
                instruments[FileDumperBase.INSTENDNAME], inplace=True,
            )

        instruments.to_csv(self.instrument_path.joinpath(
            f'{FileDumperBase.ALLINSTNAME}.txt'), header=False, sep=FileDumperBase.SEP, index=True)
    
    def dump_cal(self):
        self.calendar = (
            self.calendar if self.dump_mode != "update" else 
            sorted(list(set(self.calendar) | set(self.old_calendar)))
        )
        pd.Series(
            self.calendar,
        ).map(lambda x: x.strftime(FileDumperBase.TIMEFMT if self.freq == "day" else FileDumperBase.HFREQTIMEFMT)
        ).to_csv(
            self.calendar_path.joinpath(f'{self.freq}.txt'), 
            index=False, 
            header=False
        )
    
    def dump_feat(self):
        if self.dump_mode == "update":
            update_calendar = sorted(list(set(self.calendar) - set(self.old_calendar)))
            update_data = self.data.loc(axis=0)[:, update_calendar]
        else:
            update_data = self.data

        def _ensure_path(code):
            code_path = self.feature_path.joinpath(code.lower())
            code_path.mkdir(parents=True, exist_ok=True)
            return code_path

        def _overwrite(data):
            code_path = _ensure_path(data.index.get_level_values(0)[0])
            for feat in self.data.columns:
                np.hstack([self.calendar.index(data.index.get_level_values(1)[0]), 
                    data[feat].values]).astype('float32').tofile(
                        code_path.joinpath(f'{feat}.{self.freq}.{FileDumperBase.SUFFIX}'))
        
        def _update(data):
            code_path = _ensure_path(data.index.get_level_values(0)[0])
            for feat in self.data.columns:
                with open(code_path.joinpath(f'{feat}.{self.freq}.{FileDumperBase.SUFFIX}'), 'ab') as f: 
                    data.loc[:, feat].values.astype('float32').tofile(f)

        if update_data.empty:
            print('Your data is up-to-date!')
            return
        update_data.groupby(level=0).apply(_overwrite if self.dump_mode != "update" else _update)

    def dump(self):
        self.dump_inst()
        self.dump_cal()
        self.dump_feat()


class IndexCompDumper(FileDumperBase):
    def dump_inst(self):
        def _save(data):
            entry_date = data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).min())
            exit_date = data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).max())
            instruments = pd.concat([entry_date, exit_date], axis=1)
            instruments.applymap(lambda x: x.strftime(FileDumperBase.TIMEFMT)
                ).to_csv(
                    str(self.instrument_path.joinpath(
                        data[self.inst_field or self.date_field].iloc[0].lower() + '.txt'
                    ).resolve()),
                    sep = FileDumperBase.SEP, header = False,
                )
        self.data.groupby(self.inst_field or self.date_field).apply(_save)


    def dump_cal(self):
        pass

    def dump_feat(self):
        pass

    def dump(self):
        self.dump_inst()
    

if __name__ == "__main__":
    dumper = IndexCompDumper(
        file_path = "data/index-weights/each-index",
        file_type = "parquet",
        date_field = "index",
        inst_field = "index",
        dump_field = None,
        dump_path = "./data/qlib-day",
        dump_mode = "overwrite",
        name_pattern = '.*',
        freq = "day",
    )
    dumper.dump()
