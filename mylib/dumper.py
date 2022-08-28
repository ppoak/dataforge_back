import struct
import numpy as np
import pandas as pd
from pathlib import Path


class Dumper:
    _DATE_LEV = 0
    _INST_LEV = 1
    _SEP = '\t'
    _DATE_FMT = '%Y-%m-%d'
    _DATETIME_FMT = '%Y-%m-%d %H:%M:%S'
    _FEAT_DIR = 'features'
    _CAL_DIR = 'calendars'
    _INST_DIR = 'instruments'
    
    def __init__(
        self,
        data: 'pd.DataFrame | pd.Series',
        *,
        date_col: str = None,
        inst_col: str = None,
        uri: str = './qlib_day',
        mode: str = "w",
        freq: str = "day",
    ):
        """NOTE: There are only 3 available mode: w, a, f. 'w' mode means
        write, meaning to overwrite all information in the given uri; 'a'
        means to consider the given uri is a database, update the new dates
        and instrument information; 'f' mode means add another feature,
        only apply this when new feature is to be dumped.
        """
        self._date_col = date_col
        self._inst_col = inst_col
        self._mode = mode
        self._data = data
        self._freq = freq
        self._process_data()
        self._uri = Path(uri)
        
        self._uri.mkdir(parents=True, exist_ok=True)
        self._feat_dir = Path(self._uri).joinpath(Dumper._FEAT_DIR)
        self._cal_dir = Path(self._uri).joinpath(Dumper._CAL_DIR)
        self._inst_dir = Path(self._uri).joinpath(Dumper._INST_DIR)
        self._feat_dir.mkdir(parents=True, exist_ok=True)
        self._cal_dir.mkdir(parents=True, exist_ok=True)
        self._inst_dir.mkdir(parents=True, exist_ok=True)
        
        if self._mode in list('af'):
            self._old_cal = pd.DatetimeIndex(pd.read_csv(
                self._cal_dir.joinpath(f"{self._freq}.txt"
            ), header=None, parse_dates=[0]).iloc[:, 0])
            self._old_inst = pd.read_csv(
                self._inst_dir.joinpath(f"all.txt"
            ), header=None, parse_dates=[1, 2], index_col=[0], sep=Dumper._SEP)
    
    def _process_data(self):
        if isinstance(self._data, pd.Series) and not (self._date_col is None and self._inst_col is None):
            raise ValueError("When passing a series, you must set date_col and "
                "instrument_col to None and ensure they are index")

        for col in [self._date_col, self._inst_col]:
            if col is not None:
                self._data.set_index(col, append=True, inplace=True)
        
        if len(self._data.index.levels) == 3:
            self._data = self._data.droplevel(0)
        
        # ensure the datetime index format
        self._data.index = pd.MultiIndex.from_arrays([
            pd.to_datetime(self._data.index.get_level_values(Dumper._DATE_LEV), errors='ignore'),
            pd.to_datetime(self._data.index.get_level_values(Dumper._INST_LEV), errors='ignore'),
        ])
        if not isinstance(self._data.index.levels[Dumper._DATE_LEV], pd.DatetimeIndex):
            self._data = self._data.swaplevel()
        if isinstance(self._data, pd.Series):
            self._data = self._data.to_frame()
            
        return self
    
    def _dump_cal(self):
        self._cal = self._data.index.levels[0]

        if self._mode == 'a':
            self._cal = self._cal.union(self._old_cal)
            self._cal = self._cal[self._cal >= self._old_cal.min()]
            self._update_cal = self._cal.difference(self._old_cal)
        
        elif self._mode == 'f':
            if (self._cal > self._old_cal.max()).any():
                print("Please do not dump any future data using 'f' mode, you "
                    "can update database using 'a' mode first then use 'f' mode")
            return self
            
        self._cal.to_frame().to_csv(
            self._cal_dir.joinpath(f'{self._freq}.txt'),
            sep=Dumper._SEP, index=False, header=False)

        return self
    
    def _dump_inst(self):
        self._inst = self._data.index.levels[1]
        self._inst = self._data.groupby(level=Dumper._INST_LEV).apply(
            lambda x: pd.Series([
                x.index.get_level_values(Dumper._DATE_LEV).min(),
                x.index.get_level_values(Dumper._DATE_LEV).max(),
            ])
        )
        self._inst.columns = [1, 2]

        if self._mode == 'a':
            self._update_inst = self._inst.index.difference(self._old_inst.index)
            leak_inst = self._old_inst.index.difference(self._inst.index)
            common_inst = self._inst.index.intersection(self._old_inst.index)
            total_inst = self._inst.index.union(self._old_inst.index)
            inst = pd.DataFrame(index=total_inst, columns=self._old_inst.columns)
            inst.loc[leak_inst] = self._old_inst.loc[leak_inst]
            inst.loc[common_inst, self._old_inst.columns[0]] = \
                self._old_inst.loc[common_inst, self._old_inst.columns[0]]
            inst.loc[common_inst, self._old_inst.columns[-1]] = \
                self._inst.loc[common_inst, self._old_inst.columns[-1]]
            inst.loc[self._update_inst] = self._inst.loc[self._update_inst]
            inst.loc[self._update_inst] = inst.loc[self._update_inst].clip(
                self._cal.min(), self._cal.max()
            )
            self._inst = inst
        
        elif self._mode == 'f':
            self._update_inst = slice(None)
            new_inst = self._inst.index.difference(self._old_inst.index)
            inst = pd.concat([self._old_inst, self._inst.loc[new_inst]])
            inst.loc[new_inst, inst.columns[0]] = inst.loc[new_inst, 
                inst.columns[0]].clip(self._cal.min(), self._cal.max())
            inst.loc[new_inst, inst.columns[-1]] = inst.loc[new_inst,
                inst.columns[-1]].clip(self._cal.min(), self._cal.max())
            self._inst = inst
            
        self._inst.applymap(
            lambda x: x.strftime(Dumper._DATE_FMT if self._freq == 'day' else Dumper._DATETIME_FMT
        )).to_csv(self._inst_dir.joinpath(f'all.txt'), sep=Dumper._SEP, header=False, index=True)
        
        return self
    
    def _dump_feat(self):
        cal_list = self._cal.sort_values().to_list()

        def _ensure_path(code):
            code_path = self._feat_dir.joinpath(code.lower())
            code_path.mkdir(parents=True, exist_ok=True)
            return code_path
        
        def _overwrite(data):
            code_path = _ensure_path(data.index.get_level_values(Dumper._INST_LEV)[0])
            for feat in data.columns:
                np.hstack([
                    cal_list.index(data[feat].index.get_level_values(Dumper._DATE_LEV).min()), 
                    data[feat].values
                ]).astype('<f').tofile(code_path.joinpath(f'{feat}.{self._freq}.bin'))
        
        def _update(data):
            code_path = _ensure_path(data.index.get_level_values(Dumper._INST_LEV)[0])
            for feat in data.columns:
                with open(code_path.joinpath(f'{feat}.{self._freq}.bin'), 'ab') as f: 
                    data.loc[:, feat].values.astype('<f').tofile(f)
            
        update_inst = getattr(self, '_update_inst', None)
        update_cal = getattr(self, '_update_cal', None)
        if isinstance(update_inst, slice) or (update_inst is not None and not update_inst.empty):
            self._data.loc(axis=0)[self._cal.min():self._cal.max(), 
                update_inst].groupby(level=Dumper._INST_LEV).apply(_overwrite)
            
        if update_cal is not None and not update_cal.empty:
            self._data.loc(axis=0)[update_cal, self._old_inst.index].groupby(level=Dumper._INST_LEV).apply(_update)
        
        if update_inst is None and update_cal is None:
            self._data.groupby(level=Dumper._INST_LEV).apply(_overwrite)

    def dump(self):
        self._dump_cal()._dump_inst()._dump_feat()


class IndexCompDumper(Dumper):

    def __init__(
        self,
        data: str,
        *,
        date_col: str = None,
        inst_col: str = None,
        index_col: str = None,
        uri: str = "./qlib_day",
    ):
        assert index_col is not None, "Index field cannot be None, use index level like 0 instead"
        self._index_col = index_col
        super().__init__(
            data,
            date_col=date_col,
            inst_col=inst_col,
            uri=uri,
            mode='w',
            freq='day',
        )
    
    def _process_data(self):
        if isinstance(self._index_col, int):
            self._index_col = self._data.index.names[self._index_col] or f'level_{self._index_col}'
            self._data.reset_index(level=self._index_col, inplace=True)
        
        super()._process_data()        
        return self
    
    def _dump_inst(self):
        def _save(data):
            inst = data.groupby(level=Dumper._INST_LEV).apply(
                lambda x: pd.Series([
                    x.index.get_level_values(Dumper._DATE_LEV).min(),
                    x.index.get_level_values(Dumper._DATE_LEV).max(),
            ]))
            inst.applymap(
                lambda x: x.strftime(Dumper._DATE_FMT)).to_csv(
                    str(self._inst_dir.joinpath(
                        data[self._index_col].iloc[0].lower() + '.txt'
                    ).resolve()),
                    sep = Dumper._SEP, header = False,
                )
        self._data.groupby(self._index_col).apply(_save)
        return self
    
    def dump(self):
        self._dump_inst()


class PDumper:

    _IDX_TYPE = 'I'
    _PERIOD_TYPE = 'I'
    _ANN_TYPE = 'I'
    _VALUE_TYPE = 'd'
    _NEXT_TYPE = 'I'
    _ROW_SIZE = struct.calcsize(''.join([
        _PERIOD_TYPE, _ANN_TYPE,
        _VALUE_TYPE, _NEXT_TYPE,
    ]))
    _NAN_ADDR = 0xFFFFFFFF
    
    def __init__(
        self,
        data: 'pd.DataFrame | pd.Seires',
        *,
        inst_col: str = None,
        ann_col: str = None,
        period_col: str = None,
        uri: str = './qlib_day',
        mode: str = "w",
        freq: str = "q",
    ):
        """NOTE: Now only 'w' mode is supported, evertime we just
        pouring all data into database and replace all existing data.
        """
        self._ann_col = ann_col
        self._period_col = period_col
        self._inst_col = inst_col
        self._mode = mode
        self._data = data
        self._freq = freq
        self._uri = Path(uri)
        self._financial_dir = self._uri.joinpath('financial')
        self._process_data()
        self._financial_dir.mkdir(parents=True, exist_ok=True)

    def _process_data(self):
        # 1. We don't allow any meaningful index, for it's unnecessary
        # 2. format the date related column to datetime64[ns]
        # 3. converting the value columns to float64
        assert (self._inst_col is not None and self._ann_col is not None
            and self._period_col is not None), "You should use data without index"
        
        type_dict = {
            self._period_col: "datetime64[ns]",
            self._ann_col: "datetime64[ns]",
            self._inst_col: "object"
        }
        type_dict.update(dict(zip(
            self._data.columns.difference([self._period_col, self._ann_col, self._inst_col]),
            ["float64" for _ in range(self._data.columns.size - 3)]
        )))
        self._data = self._data.astype(type_dict)
        
    def _dump(self, data):
        """Assuming data we get here have three columns, report_date, ann_date and values"""

        def _save(d):
            feat_name = d.index.get_level_values(-1)[0]
            d = d.droplevel(-1).reset_index()
            d[self._period_col] = d[self._period_col].map(
                lambda x: x.year * 100 + x.month // 4 + 1)
            d[self._ann_col] = d[self._ann_col].map(
                lambda x: x.year* 10000 + x.month * 100 + x.day)
            dp = d[self._period_col].duplicated(keep='last')
            d['_next'] = ~dp * PDumper._NAN_ADDR
            d.loc[dp, '_next'] = (d[dp].index + 1) * PDumper._ROW_SIZE
            d = d.loc[:, [self._ann_col, self._period_col, feat_name, '_next']].astype({
                self._ann_col: PDumper._ANN_TYPE,
                self._period_col: PDumper._PERIOD_TYPE,
                feat_name: PDumper._VALUE_TYPE,
                '_next': PDumper._NEXT_TYPE,
            })
            np.array(list(map(tuple, d.values)), dtype = [
                ('date', PDumper._ANN_TYPE),
                ('period', PDumper._PERIOD_TYPE),
                ('value', PDumper._VALUE_TYPE),
                ('_next', PDumper._NEXT_TYPE)
            ]).tofile(code_path.joinpath(f'{feat_name.lower()}_{self._freq}.data'))
            i = np.hstack([
                d[self._period_col].min() // 100, *[PDumper._NAN_ADDR] * (d[self._period_col].min() % 100 - 1),
                d[~d[self._period_col].duplicated(keep='first')].index * PDumper._ROW_SIZE, *[PDumper._NAN_ADDR]
                * (4 - d[self._period_col].max() % 100)]).astype(PDumper._IDX_TYPE)
            i.tofile(code_path.joinpath(f'{feat_name.lower()}_{self._freq}.index'))
            
        # 1. transforming the report_col to a non-leaping date_range, then enter _save function
        # 2. calculating _next col and constructing the index data
        # 3. transform data into corresponding format and save
        code = data[self._inst_col].iloc[0]
        code_path = self._financial_dir.joinpath(code.lower())
        code_path.mkdir(parents=True, exist_ok=True)
        data = data.drop(self._inst_col, axis=1)

        data = data.set_index(self._period_col)
        duplicated = data.index.duplicated(keep='first')
        nodudata: pd.DataFrame = data[~duplicated]
        nodudata = nodudata.reindex(pd.date_range(
            nodudata.index.min(), nodudata.index.max(), freq='q'))
        nodudata.index.name = self._period_col
        data = pd.concat([nodudata, data.loc[duplicated]]).sort_index()
        data = data.ffill().dropna(axis=0, subset=self._ann_col)
        data = data.set_index(self._ann_col, append=True).stack()

        data.groupby(level=-1).apply(_save)

    def dump(self):
        self._data.groupby(self._inst_col).apply(self._dump)
