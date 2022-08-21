import os
import numpy as np
import pandas as pd
from pathlib import Path


class SingleFileDumper:
    SEP = '\t'
    SUFFIX = '.bin'
    FREQ = 'day'
    IDXPRESENT = 'index'
    ALLINSTNAME = 'all'
    INSTPATH = 'instruments'
    CALPATH = 'calendars'
    FEATPATH = 'features'

    def __init__(
        self,
        file_path: str,
        file_type: str,
        date_field: str,
        inst_field: str,
        dump_path: str,
    ):
        data_reader = eval(f'pd.read_{file_type}')
        data = data_reader(file_path)
        if inst_field == SingleFileDumper.IDXPRESENT and date_field == SingleFileDumper.IDXPRESENT:
            if not isinstance(data.index.levels[1], pd.DatetimeIndex):
                data = data.swaplevel()
        elif inst_field == SingleFileDumper.IDXPRESENT and date_field != SingleFileDumper.IDXPRESENT:
            data.index = pd.MultiIndex.from_arrays([data.index, data[date_field]])
        elif inst_field != SingleFileDumper.IDXPRESENT and date_field == SingleFileDumper.IDXPRESENT:
            data.index = pd.MultiIndex.from_arrays([data[inst_field], data.index])
        else:
            data = data.set_index([inst_field, date_field])
        self.data = data

        self.instruments = set(data.index.levels[0])
        self.calendar = sorted(list(set(data.index.levels[1])))
        self.dump_path = Path(dump_path)
        self.dump_path.mkdir(parents=True, exist_ok=True)
        self.instrument_path = Path(self.dump_path).joinpath(SingleFileDumper.INSTPATH)
        self.calendar_path = Path(self.dump_path).joinpath(SingleFileDumper.CALPATH)
        self.feature_path = Path(self.dump_path).joinpath(SingleFileDumper.FEATPATH)
        self.instrument_path.mkdir(parents=True, exist_ok=True)
        self.calendar_path.mkdir(parents=True, exist_ok=True)
        self.feature_path. mkdir(parents=True, exist_ok=True)
    
    def dump_inst(self):
        entry_date = self.data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).min())
        exit_date = self.data.groupby(level=0).apply(lambda x: x.index.get_level_values(1).max())
        instruments = pd.concat([entry_date, exit_date], axis=1)
        instruments.to_csv(self.instrument_path. joinpath(f'{SingleFileDumper.ALLINSTNAME}.txt'), header=False, sep=SingleFileDumper.SEP, index=True)
    
    def dump_cal(self):
        pd.Series(
            self.calendar,
        ).to_csv(self.calendar_path.joinpath(f'{SingleFileDumper.FREQ}.txt'), index=False, header=False)
    
    def dump_feat(self):
        dump_features = set(self.data.columns)

        def _dump(data):
            code = data.index.get_level_values(0)[0]
            code_path = self.feature_path.joinpath(code.lower())
            code_path.mkdir(parents=True, exist_ok=True)
            for feat in dump_features:
                np.hstack([self.calendar.index(data.index.get_level_values(1)[0]), 
                    data[feat].values]).astype('float32').tofile(code_path.joinpath(f'{feat}.{SingleFileDumper.FREQ}.{SingleFileDumper.SUFFIX}'))
        
        self.data.groupby(level=0).apply(_dump)
        
    def dump(self):
        self.dump_inst()
        self.dump_cal()
        self.dump_feat()
