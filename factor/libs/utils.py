import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class RollingTrain:
    """
    This RollingTrain class is specifically designed for
    factor machine learning. In order to make full use of 
    data, we avoiding using valid set. Instead, we use the 
    earliest sample (is) in dataset to learn some parameters for 
    following training (oos).
    """

    def __init__(
        self,
        min_days: int = 20,
        max_days: int = 40,
        pred_days: int = 5,
        learn_days: int = 200,
        exp_path: str = 'data/intermediate/results/',
        exp_name: str = 'test'
    ) -> None:
        """A class used for rolling training
        ------------------------------------

        handler: qlib.data.dataset.handler.DataHandlerLP,
            a data handler constructed by loader
        min_days: int, minimum days in training dataset
        max_days: int, maximum days in training dataset
        valid_days: int, the days contained in valid set
        pred_days: int, the days contained in predict set
        learn_days: int, the day period in the whole dataset
            for some parameters learning, e.g. best epoch
        """
        self.min_days = min_days
        self.max_days = max_days
        self.pred_days = pred_days
        self.learn_days = learn_days
        self.exp_path = Path(exp_path)
        self.exp_name = self.exp_path.joinpath(exp_name)
        self.exp_name.mkdir(parents=True, exist_ok=True)
        self.exp_name.joinpath('oos').mkdir(parents=True, exist_ok=True)

    def rolling(self, model, dataset, **kwargs):
        """This method rolls on the dataset
        ---------------------------------------

        model: a pre-initialized model instance, 
            and the fit, predict method should be implemented
        kwargs: other keyword arguments applies to model.fit method
        """
        datetime_index = dataset.index.levels[0]
        params = []
        for i, idx in list(enumerate(datetime_index))[::self.pred_days]:
            print(f"{'=' * 20} Day {idx.strftime('%Y-%m-%d')} {'=' * 20}")
            if i < self.min_days + self.pred_days - 1:
                continue
            pred_end_idx = i
            pred_start_idx = i - self.pred_days + 1
            train_end_idx = pred_start_idx - 1
            train_start_idx = max(min(train_end_idx - self.min_days, train_end_idx - self.max_days), 0)
            
            train = dataset.loc[datetime_index[train_start_idx]:datetime_index[train_end_idx]]
            test = dataset.loc[datetime_index[pred_start_idx]:datetime_index[pred_end_idx]]
            
            if i <= self.learn_days:
                model.fit(train, test, **kwargs)
                max_pos = np.argmax(model.evals_result['valid']['top_ret'])
                params.append(max_pos + 1)
                print(f'There are {max_pos+1} iters in best model')
            else:
                model.fit(train, valid=None, force_iter=int(np.mean(params)), **kwargs)

            top_ret = (max(model.evals_result['valid']['top_ret']) if model.evals_result is not None else
                model.ret.loc[model.predict(test).groupby(level=0).apply(
                    lambda x: x.sort_values(ascending=False).iloc[:int(len(x) * model.top)]).droplevel(0).index].
                    groupby(level=0).mean().squeeze().sum())
            
            print(f"[Rolling] Summation Top Return: {top_ret:.4f}")

            pred = model.predict(test)
            filename = ("oos/" if i > self.learn_days else "") + "pred_{}_{}".format(
                datetime_index[pred_start_idx].strftime('%Y%m%d'), 
                datetime_index[pred_end_idx].strftime('%Y%m%d')
            )
            pred.to_frame(name='score').to_parquet(self.exp_name.joinpath(filename + '.parquet'))


class TorchDataset(Dataset):
    def __init__(self, df) -> None:
        self.feature = df['feature'].values
        self.label = df['label'].values
        self.length = len(df)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.feature[index], self.label[index]


class PreProcessor:
    """This processor only supports for single factor pre-process 
    for the multi-factor processing takes too much time
    """

    def __init__(
        self,
        data: 'pd.DataFrame | pd.Series',
        level: int = None,
    ) -> None:
        """
        data: pd.DataFrame, the wide form single factor matrix,
            pd.Series, the multi-index form factor column
        level: int, the level representing data
        """
        self.data = data
        self.level = level

    @staticmethod
    def sigma(data: pd.DataFrame, dev: int = 3):
        """3-sigma deextreme method
        ----------------------------

        data: pd.DataFrame, the unprocessed data
        dev: int, pull deviation larger than dev to dev
        """
        data_mean = data.mean()
        data_std = data.std()
        return data.clip(data_mean - dev * data_std, data_mean + dev * data_std, axis=0)
    
    @staticmethod
    def minmax(data: pd.DataFrame):
        """Min Max normalization method
        -------------------------------

        data: pd.DataFrame, the unprocessed data
        """
        data_max = data.max()
        data_min = data.min()
        return (data - data_min) / (data_max - data_min)
    
    @staticmethod
    def mad(data: pd.DataFrame, dev: int = 5):
        """MAD deextreme method
        ------------------------

        data: pd.DataFrame, the unprocessed data
        dev: int, pull deviation larger than dev to dev
        """
        data_med = data.median()
        data_mad = (data - data_med).abs().median()
        return data.clip(data - dev * data_mad, data + dev * data_mad, axis=0)

    @staticmethod
    def zscore(data: 'pd.DataFrame | pd.Series'):
        """ZScore normalization
        ------------------------

        data: pd.DataFrame, the unprocessed data        
        """
        return (data - data.mean()) / data.std()

    @property
    def result(self):
        return self.data

    def __call__(self, method: str = 'sigma', **kwargs):
        method = getattr(self, method)
        if self.level is None:
            self.data = method(self.data, **kwargs)
        else:
            self.data = method(self.data.unstack(level=1 - self.level), **kwargs)
        return self
