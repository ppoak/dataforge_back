import qlib
import pandas as pd
from pathlib import Path
from qlib.data.dataset import DatasetH
from qlib.contrib.model.pytorch_nn import DNNModelPytorch
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader


qlib.init(provider_uri = './data/qlib_day')
data = pd.read_parquet('./data/intermediate/feature_info/normalized_dataset.parquet')
label = pd.read_parquet('./data/intermediate/forward_return/1d_vwap_vwap.parquet')
label.columns = ['label']


class StaticHandler(DataHandlerLP):
    def __init__(
        self, 
        data,
        instruments = None, 
        start_time = None, 
        end_time = None, 
        drop_raw = False, 
        **kwargs
    ):
        data_loader = StaticDataLoader(config = data)
        super().__init__(
            instruments = instruments, 
            start_time = start_time, 
            end_time = end_time, 
            data_loader = data_loader, 
            drop_raw = drop_raw, 
            **kwargs
        )


class RollingTrain:
    def __init__(
        self,
        handler, 
        min_days: int = 20,
        max_days: int = 40,
        valid_days: int = 10,
        pred_days: int = 5,
        exp_name: str = 'lgbm'
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
        self.handler = handler
        self.min_days = min_days
        self.max_days = max_days
        self.valid_days = valid_days
        self.pred_days = pred_days
        self.exp_name = Path(f'./data/intermediate/results/{exp_name}')
        self.exp_name.mkdir(parents=True, exist_ok=True)

    def rolling(self, model, **kwargs):
        """This method rolls on the datahandler
        ---------------------------------------

        model: a pre-initialized model instance, 
            and the fit, predict method should be implemented
        kwargs: other keyword arguments applies to model.fit method
        """
        datetime_index = self.handler.fetch(data_key='infer', col_set='label').index.levels[0]
        for i, idx in enumerate(datetime_index):
            print(f'{"-" * 20} Training for {idx.strftime("%Y-%m-%d")} {"-" * 20}')
            if i < self.min_days + self.valid_days + self.pred_days - 1:
                continue
            pred_end_idx = i
            pred_start_idx = i - self.pred_days + 1
            valid_end_idx = pred_start_idx - 1
            valid_start_idx = valid_end_idx - self.valid_days + 1
            train_end_idx = valid_start_idx - 1
            train_start_idx = max(min(train_end_idx - self.min_days, train_end_idx - self.max_days), 0)
            
            dataset = DatasetH(handler=self.handler, segments={
                "train": (datetime_index[train_start_idx], datetime_index[train_end_idx]),
                "valid": (datetime_index[valid_start_idx], datetime_index[valid_end_idx]),
                "test": (datetime_index[pred_start_idx], datetime_index[pred_end_idx]),
            })

            model.fit(dataset, **kwargs)
            pred = model.predict(dataset, segment='test')
            label_ = label.loc[pred.index]
            pred_label = pd.concat([pred, label_], axis=1)
            pred_label.columns = ['score', 'label']
            filename = "pred_label_{}_{}".format(
                datetime_index[pred_start_idx].strftime('%Y%m%d'), 
                datetime_index[pred_end_idx].strftime('%Y%m%d')
            )
            pred_label.to_pickle(self.exp_name.joinpath(filename))

handler = StaticHandler(data)
RollingTrain(
    handler, 
    min_days=100, 
    max_days=120, 
    valid_days=20, 
    pred_days=10,
    exp_name='dnn'
).rolling(
    DNNModelPytorch(
        lr = 8e-2,
        lr_decay = 0.3,
        lr_decay_steps = 100,
        optimizer = 'adam',
        max_steps = 4000,
        batch_size = 500,
        GPU = 0,
        weight_decay = 4e-4,
        pt_model_kwargs = {
            'input_dim': 85,
            'layers': (512, 512)
        },
    )
)