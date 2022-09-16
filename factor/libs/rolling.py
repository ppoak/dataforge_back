import numpy as np
import pandas as pd
from pathlib import Path
from factor.libs.models import DNNModel


class RollingTrain:
    def __init__(
        self,
        min_days: int = 20,
        max_days: int = 40,
        # valid_days: int = 10,
        pred_days: int = 5,
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
        """
        self.min_days = min_days
        self.max_days = max_days
        # self.valid_days = valid_days
        self.pred_days = pred_days
        self.exp_name = Path(f'data/intermediate/results/{exp_name}')
        self.exp_name.mkdir(parents=True, exist_ok=True)
        self.exp_name.joinpath('oos').mkdir(parents=True, exist_ok=True)

    def rolling(self, model, dataset, learn_epoch, **kwargs):
        """This method rolls on the datahandler
        ---------------------------------------

        model: a pre-initialized model instance, 
            and the fit, predict method should be implemented
        kwargs: other keyword arguments applies to model.fit method
        """
        # datetime_index = self.handler.fetch(data_key='infer', col_set='label').index.levels[0]
        datetime_index = dataset.index.levels[0]
        epochs = []
        for i, idx in list(enumerate(datetime_index))[::self.pred_days]:
            print(f"{'=' * 20} Day {idx.strftime('%Y-%m-%d')} {'=' * 20}")
            if i < self.min_days + self.pred_days - 1:
                continue
            pred_end_idx = i
            pred_start_idx = i - self.pred_days + 1
            # valid_end_idx = pred_start_idx - 1
            # valid_start_idx = valid_end_idx - self.valid_days + 1
            train_end_idx = pred_start_idx - 1
            train_start_idx = max(min(train_end_idx - self.min_days, train_end_idx - self.max_days), 0)
            
            train = dataset.loc[datetime_index[train_start_idx]:datetime_index[train_end_idx]]
            test = dataset.loc[datetime_index[pred_start_idx]:datetime_index[pred_end_idx]]
            
            if i <= learn_epoch:
                results = model.fit(train, test)
                epochs.append(len(results['train']['loss']))
            else:
                results = model.fit(train, train, max_epoch=int(np.mean(epochs)))

            pred = model.predict(test)
            # selected = pred.groupby(level=0).apply(lambda x: 
            #     x.sort_values(ascending=False).iloc[:int(len(x)*top)]).droplevel(0).index
            # top_ret = ret.loc[selected].groupby(level=0).mean()
            filename = ("oos/" if i > learn_epoch else "") + "pred_{}_{}".format(
                datetime_index[pred_start_idx].strftime('%Y%m%d'), 
                datetime_index[pred_end_idx].strftime('%Y%m%d')
            )
            pred.to_frame(name='score').to_parquet(self.exp_name.joinpath(filename + '.parquet'))


if __name__ == "__main__":
    data = pd.read_parquet('data/intermediate/feature_info/normalized_dataset.parquet')
    ret = pd.read_parquet('data/intermediate/forward_return/1d_open_open.parquet')
    RollingTrain(
        min_days = 100,
        max_days = 120,
        pred_days = 20,
        exp_name = 'onlypred_test'
    ).rolling(
        DNNModel(
            ret = ret,
            optimizer_kwargs={
                "weight_decay": 0.001,
                "lr": 0.01,
            },
            epoch = 70,
            batch_size=1000,
            ret_stop = 20,
        ),
        dataset=data,
        learn_epoch=200,
    )
