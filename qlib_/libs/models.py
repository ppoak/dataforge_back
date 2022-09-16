import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, df) -> None:
        self.feature = df['feature'].values
        self.label = df['label'].values
        self.length = len(df)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.feature[index], self.label[index]


class DNNModel:
    
    def __init__(
        self,
        ret,
        ret_stop = 5,
        top = 0.1,
        dnn_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        scheduler=None,
        scheduler_kwargs=None,
        loss=None,
        loss_kwargs=None,
        epoch=300,
        batch_size=2000,
        early_stop_rounds=50,
        eval_steps=20,
        GPU=0,
        seed=None,
    ) -> None:
        self.ret = ret
        self.ret_stop = ret_stop
        self.top = top
        self.seed = seed
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.early_stop_rounds = early_stop_rounds
        self.best_step = None
        if isinstance(GPU, str):
            self.device = torch.device(GPU)
        else:
            self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if dnn_kwargs is None:
            dnn_kwargs = {
                "input_dim": 85,
                "layers": (512, 512),
                "output_dim": 1,
                "act": "LeakyReLU",
                "act_kwargs": {
                    "negative_slope": 0.1, 
                    "inplace": False,
                }
            }
        self.dnn_model = DNN(**dnn_kwargs)
        self.dnn_model.to(self.device)
        print(self.dnn_model)

        if optimizer is None:
            optimizer = "SGD"
        if optimizer_kwargs is None:
            optimizer_kwargs = {
                "lr": 0.001,
                "weight_decay": 0.001
            }
        optimizer_kwargs.update({
            "params": self.dnn_model.parameters()
        })
        self.optimizer: torch.optim.Optimizer = getattr(
            torch.optim, optimizer)(**optimizer_kwargs)
        
        if loss is None:
            loss = "MSELoss"
        if loss_kwargs is None:
            loss_kwargs = {
                "reduction": "mean"
            }
        self.loss: torch.nn.MSELoss = getattr(nn, loss)(**loss_kwargs)

        if scheduler is None:
            scheduler = "ReduceLROnPlateau"
        if scheduler_kwargs is None:
            scheduler_kwargs = {
                "optimizer": self.optimizer,
                "mode": "min",
                "factor": 0.5,
                "patience": 10,
                "verbose": True,
                "threshold": 0.0001,
                "threshold_mode": "rel",
                "cooldown": 0,
                "min_lr": 0.00001,
                "eps": 1e-8,
            }
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = getattr(
            torch.optim.lr_scheduler, scheduler)(**scheduler_kwargs)
    
    def train_epoch(
        self,
        train,
    ):
        for i, (feature, label) in enumerate(train):
            self.dnn_model.train()
            pred = self.dnn_model(feature)
            loss = self.loss(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % self.eval_steps == 0 or (i + 1) == len(train):
                print(f"[Batch {i + 1}] loss on train {loss.item()}")

    def test_epoch(
        self,
        test
    ):
        self.dnn_model.eval()
        with torch.no_grad():
            preds = self.dnn_model(torch.from_numpy(test['feature'].values))
            test_loss = self.loss(preds, torch.from_numpy(test['label'].values))
            preds_df = pd.Series(preds.numpy().reshape(-1), index=test.index)
            select_inst = preds_df.groupby(level=0).apply(
                lambda x: x.sort_values(ascending=False).iloc[:int(len(x) * self.top)]
            ).droplevel(0).index
            top_ret = self.ret.loc[select_inst].groupby(level=0).mean()
            print(f"[Epoch] Loss: {test_loss.item():.4f}, "
                f"Top Portfolio Mean Return {top_ret.sum().iloc[0]:.4f}")
        return test_loss.item(), top_ret.sum().iloc[0]

    def fit(
        self,
        train,
        test,
        max_epoch = np.inf,
    ):
        results = {
            "train": {"loss": [], "top_ret": []}, 
            "test": {"loss": [], "top_ret": []},
        }
        traindataset = DataLoader(DataSet(train), batch_size=self.batch_size, shuffle=True)
        best_ret = -np.inf
        ret_stop = 0
        for epoch in range(1, min(self.epoch, max_epoch) + 1):
            if ret_stop > self.ret_stop:
                print(f"Top return didn't improve for {self.ret_stop} epoch, now quitting")
                break
            print(f"{'-' * 20} Epoch {epoch} {'-' * 20}")
            self.train_epoch(traindataset)
            train_loss, train_top_ret = self.test_epoch(train)
            test_loss, test_top_ret = self.test_epoch(test)

            if test_top_ret > best_ret:
                best_ret = test_top_ret
                best_param = deepcopy(self.dnn_model.state_dict())
                ret_stop = 0
            else:
                ret_stop += 1
            
            self.scheduler.step(test_top_ret.sum())
            
            results["train"]["loss"].append(train_loss)
            results["train"]["top_ret"].append(train_top_ret)
            results["test"]["loss"].append(test_loss)
            results["test"]["top_ret"].append(test_top_ret)
        
        self.dnn_model.load_state_dict(best_param)
        return results
    
    def predict(self, test):
        self.dnn_model.eval()
        with torch.no_grad():
            pred = self.dnn_model(torch.from_numpy(test['feature'].values))
            pred = pd.Series(pred.detach().numpy().reshape(-1), index=test.index)
        return pred


class DNN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        layers: tuple = (512,),
        act: str = "LeakyReLU",
        act_kwargs: dict = None,
    ):
        super(DNN, self).__init__()
        act_kwargs = act_kwargs or {}
        layers = [input_dim] + list(layers)
        dnn_layers = []
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        hidden_units = input_dim

        for _input_dim, hidden_units in zip(layers[:-1], layers[1:]):
            fc = nn.Linear(_input_dim, hidden_units)
            activation = getattr(nn, act)(**act_kwargs)
            bn = nn.BatchNorm1d(hidden_units)
            seq = nn.Sequential(fc, bn, activation)
            dnn_layers.append(seq)
        
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        fc = nn.Linear(hidden_units, output_dim)
        dnn_layers.append(fc)
        self.dnn_layers = nn.ModuleList(dnn_layers)
        self._weight_init()
    
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
            
    def forward(self, x):
        cur_output = x
        for now_layer in self.dnn_layers:
            cur_output = now_layer(cur_output)
        return cur_output


if __name__ == "__main__":
    data = pd.read_parquet('data/intermediate/feature_info/normalized_dataset.parquet')
    ret = pd.read_parquet('data/intermediate/forward_return/1d_open_open.parquet')
    train = data.loc["2018-01-01":"2018-03-31"]
    test = data.loc["2018-04-01":"2018-04-20"]
    model = DNNModel(
        ret = ret,
        optimizer_kwargs={
            "weight_decay": 0.001,
            "lr": 0.01,
        },
        epoch = 50,
        batch_size=1000,
        ret_stop = 20,
    )
    results = model.fit(train, test)
    
    import matplotlib.pyplot as plt
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
    axes[0][0].plot(results['train']['loss'])
    axes[0][1].plot(results['train']['top_ret'])
    axes[1][0].plot(results['test']['loss'])
    axes[1][1].plot(results['test']['top_ret'])
    plt.savefig('test.png')
