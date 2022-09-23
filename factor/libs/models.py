import abc
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import lightgbm as lgb
from tqdm import tqdm
from copy import deepcopy
from operator import gt, lt
from .utils import TorchDataset
from torch.utils.data import DataLoader
from pytorch_tabnet.metrics import Metric
from lightgbm.callback import EarlyStopException
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold


class TopRetModelBase:
    def __init__(self, ret, top, ret_stop):
        self.ret = ret
        self.top = top
        self.ret_top = ret_stop
        
    @abc.abstractmethod
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame = None, **kwargs) -> 'TopRetModelBase':
        raise NotImplementedError("subclasses should implement fit method")
    
    @abc.abstractmethod
    def predict(self, test: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("subclasses should implement predict method")


class LinearModel(TopRetModelBase):
    def __init__(
        self, 
        ret,
        top = 0.1,
        ret_stop = 10,
        in_feature = 5,
        out_feature = 1,
        epoch = 100,
        batch_size = 1024,
        optimizer_cls = torch.optim.AdamW,
        optimizer_kwargs = {"lr": 1e-2, "weight_decay": 1e-2, "betas": (0.9, 0.999)},
    ):
        self.ret = ret
        self.top = top
        self.ret_stop = ret_stop
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.evals_result = None
        
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame = None, force_iter: int = None, **kwargs):
        self.model = Linear(in_features=self.in_feature, out_features=self.out_feature)
        train_dataset = DataLoader(TorchDataset(train), batch_size=self.batch_size)
        self.optimizer_kwargs.update({"params": self.model.parameters()})
        self.optimizer = self.optimizer(**self.optimizer_kwargs)
        loss_fn = nn.MSELoss(reduction='mean')
        best_score = -np.inf
        best_iter = 0
        if valid is not None:
            self.evals_result = {"valid": {"top_ret": []}}
        for epoch in range(self.epoch if force_iter is None else force_iter):
            self.model.train()
            for step, (feature, label) in enumerate(train_dataset):
                pred = self.model(feature.float())
                loss = loss_fn(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if valid is not None:
                top_ret = self.ret.loc[self.predict(valid).groupby(level=0).apply(
                    lambda x: x.sort_values(ascending=False).iloc[:int(len(x) * self.top)]
                ).droplevel(0).index].groupby(level=0).mean().squeeze().sum()
                self.evals_result['valid']['top_ret'].append(top_ret)
                print(f"[Epoch {epoch + 1}] Top Return: {top_ret}")
                if top_ret > best_score:
                    best_score = top_ret
                    best_iter = epoch
                    model_param = deepcopy(self.model.state_dict())
                if epoch - best_iter >= self.ret_stop:
                    print(f"[Early Stop] Top Return didn't imporve in {self.ret_stop} epoch, quitting")
                    print(f"[Early Stop] The best top return is {best_score} in epoch {best_iter}")
                    break
        self.model.load_state_dict(model_param)
        return self
    
    def predict(self, test: pd.DataFrame):
        self.model.eval()
        with torch.no_grad():
            return pd.Series(
                self.model(torch.from_numpy(test['feature'].values).float()).detach().numpy().reshape(-1),
                index=test.index,
            )


class Linear(nn.Module):
    def __init__(self, in_features: int = 5, out_features: int = 1):
        super(Linear, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=in_features)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.seq = nn.Sequential(self.bn, self.linear)
    
    def forward(self, x):
        return self.seq(x)


class DNNModel(TopRetModelBase):
    
    def __init__(
        self,
        ret,
        top = 0.1,
        ret_stop = 5,
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
        eval_steps=200,
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
        self.model = DNN(**dnn_kwargs)
        self.model.to(self.device)

        if optimizer is None:
            optimizer = "AdamW"
        if optimizer_kwargs is None:
            optimizer_kwargs = {
                "lr": 0.001,
                "weight_decay": 0.001
            }
        optimizer_kwargs.update({
            "params": self.model.parameters()
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
    
    def train_epoch(self, train: DataLoader, epoch: int):
        with tqdm(total=len(train)) as pbar:
            for i, (feature, label) in enumerate(train):
                pbar.set_description(f'[Epoch {epoch}]')
                self.model.train()
                pred = self.model(feature)
                loss = self.loss(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

    def test_epoch(self, test: pd.DataFrame, epoch: int, name: str = 'valid'):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.from_numpy(test['feature'].values))
            test_loss = self.loss(preds, torch.from_numpy(test['label'].values))
            preds_df = pd.Series(preds.numpy().reshape(-1), index=test.index)
            select = preds_df.groupby(level=0).apply(
                lambda x: x.sort_values(ascending=False).iloc[:int(len(x) * self.top)]
            ).droplevel(0).index
            top_ret = self.ret.loc[select].squeeze().groupby(level=0).mean()
            print(f"[Epoch {epoch}] {name} Loss: {test_loss.item():.4f}, "
                f"{name} Top Portfolio Mean Return {top_ret.sum():.4f}")
        return test_loss.item(), top_ret.sum()

    def fit(
        self, 
        train: pd.DataFrame, 
        valid: pd.DataFrame = None, 
        force_iter: int = None
    ):
        """Fit the train and test data
        -------------------------------
        
        In order to use all the available data, we don't set valid dataset.
        Instead, we can use the test dataset to verify the best model parameters,
        then use the parameter like ``epoch`` to train data
        """
        results = {
            "train": {"loss": [], "top_ret": []}, 
            "valid": {"loss": [], "top_ret": []},
        }
        train_dataset = DataLoader(TorchDataset(train), batch_size=self.batch_size, shuffle=True)
        best_ret = -np.inf
        ret_stop = 0
        for epoch in range(1, self.epoch + 1 if force_iter is None else force_iter + 1):
            if ret_stop > self.ret_stop:
                print(f"[Early Stop] Top return didn't improve "
                    f"for {self.ret_stop} epoch, best score: {best_ret:.4f}")
                break
            self.train_epoch(train_dataset, epoch)
            train_loss, train_top_ret = self.test_epoch(train, epoch, "train")
            if valid is not None:
                valid_loss, valid_top_ret = self.test_epoch(valid, epoch, "valid")
            else:
                valid_loss, valid_top_ret = train_loss, train_top_ret

            if valid_top_ret > best_ret:
                best_ret = valid_top_ret
                best_param = deepcopy(self.model.state_dict())
                ret_stop = 0
            else:
                ret_stop += 1
            
            self.scheduler.step(valid_top_ret.sum())
            
            results["train"]["loss"].append(train_loss)
            results["train"]["top_ret"].append(train_top_ret)
            results["valid"]["loss"].append(valid_loss)
            results["valid"]["top_ret"].append(valid_top_ret)
        
        self.model.load_state_dict(best_param)
        self.evals_result = results
        return self
    
    def predict(self, test: pd.DataFrame):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.from_numpy(test['feature'].values))
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


class LGBModel(TopRetModelBase):

    def __init__(
        self,
        ret,
        top = 0.1,
        ret_stop = 50,
        boosting_type = "gbdt",
        n_estimators = 1000,
        learning_rate = 0.01,
        num_leaves = 100,
        max_depth = 5,
        subsample = 0.72,
        subsample_freq = 10,
        colsample_bytree = 1,
        reg_alpha = 0.01,
        reg_lambda = 0.001,
        min_child_samples = 100,
    ) -> None:
        self.ret = ret
        self.top = top
        self.ret_stop = ret_stop
        self.boosting_type = boosting_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
    
    def metric(self, y_true, y_pred):
        y_true = self.ret.loc[self.valid_index]
        y_pred = pd.Series(y_pred, index=y_true.index)
        select = y_pred.groupby(level=0).apply(
            lambda x: x.sort_values(ascending=False).iloc[:int(len(x) * self.top)]
        ).droplevel(0).index
        ret = y_true.squeeze().loc[select].groupby(level=0).mean().sum()
        return ("top_ret", ret, True)
    
    def early_stopping_for_ret(self, stop_round):
        cmp_op = [lt, gt]
        best_score = [np.inf, -np.inf]
        best_iter = [0, 0]
        best_score_list = [(), ()]
        
        def _callback(env):
            for i, evals in enumerate(env.evaluation_result_list):
                if cmp_op[i](evals[2], best_score[i]):
                    best_score[i] = evals[2]
                    best_iter[i] = env.iteration
                    best_score_list[i] = evals
                # we only consider whether the top return rises
                if evals[1] == 'top_ret':
                    if env.iteration - best_iter[i] >= stop_round:
                        print(f"[Early Stop]: Best top return: {best_score[i]}," 
                            f" best iteration: {best_iter[i]}")
                        raise EarlyStopException(best_iter[i], [best_score_list[i]])
                    
        return _callback
    
    def print_evaluatoin(self, period: int = 50):
        def _callback(env) -> None:
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                print(f'[Tree {env.iteration + 1}]\tTop Return: {env.evaluation_result_list[1][2]}')
        _callback.order = 10  # type: ignore
        return _callback
    
    def fit(
        self, 
        train: pd.DataFrame, 
        valid: pd.DataFrame = None,
        force_iter: int = None
    ):
        if force_iter is None:
            if valid is not None:
                self.valid_index = valid.index
            trees = self.n_estimators
        else:
            trees = force_iter
        
        self.model = lgb.LGBMRegressor(
            boosting_type = self.boosting_type,
            n_estimators = trees,
            learning_rate = self.learning_rate,
            num_leaves = self.num_leaves,
            max_depth = self.max_depth,
            subsample = self.subsample,
            subsample_freq = self.subsample_freq,
            colsample_bytree = self.colsample_bytree,
            reg_alpha = self.reg_alpha,
            reg_lambda = self.reg_lambda,
            min_child_samples = self.min_child_samples,
        )
        if valid is None:
            eval_dict = {}
        else:
            eval_dict = {
                "eval_set": [(valid['feature'].values, valid['label'].values.reshape(-1))],
                "eval_names": ["valid"],
                "eval_metric": self.metric,
                "callbacks": [self.print_evaluatoin(period=50), self.early_stopping_for_ret(stop_round=self.ret_stop)]
            }
        self.model.fit(
            train['feature'].values, 
            train['label'].values.reshape(-1), 
            **eval_dict,
        )
        
        self.evals_result = self.model.evals_result_
        return self
    
    def predict(self, test):
        pred = self.model.predict(test['feature'].values)
        pred = pd.Series(pred, index=test.index)
        return pred


class TabNetModel(TopRetModelBase):

    def __init__(
        self,
        ret,
        top = 0.1,
        ret_stop = 2,
        n_d = 8,
        n_a = 8,
        n_steps = 10,
        gamma = 1.3,
        n_independent = 2,
        n_shared = 2,
        epsilon = 1e-15,
        momentum = 0.02,
        clip_value = None,
        lambda_sparse = 1e-3,
        optimizer_fn = torch.optim.Adam,
        optimizer_params = {"lr": 2e-2},
        verbose = 1,
    ):
        self.ret = ret
        self.top = top
        self.ret_stop = ret_stop
        self.model_params = {
            "n_d": n_d,
            "n_a": n_a,
            "n_steps": n_steps,
            "gamma": gamma,
            "n_independent": n_independent,
            "n_shared": n_shared,
            "epsilon": epsilon,
            "momentum": momentum,
            "clip_value": clip_value,
            "lambda_sparse": lambda_sparse,
            "optimizer_fn": optimizer_fn,
            "optimizer_params": optimizer_params,
            "verbose": verbose,
        }
    
    def metric(self):
        top = self.top
        ret = self.ret
        index = self.index
        class _TopRet(Metric):
            def __init__(self):
                self._name = 'top_ret'
                self._maximize = True
                self.top = top
                self.ret = ret
                self.index = index
            def __call__(self, y_true, y_pred):
                y_true = self.ret.loc[index]
                y_pred = pd.Series(y_pred.reshape(-1), index=y_true.index)
                select = y_pred.groupby(level=0).apply(
                    lambda x: x.sort_values(ascending=False).iloc[:int(len(x) * self.top)]
                ).droplevel(0).index
                ret = y_true.squeeze().loc[select].groupby(level=0).mean().sum()
                return ret
        
        return _TopRet
        
    def fit(
        self, 
        train: pd.DataFrame, 
        valid: pd.DataFrame = None, 
        force_iter: int = None, 
        **kwargs,
    ):        
        if force_iter is not None:
            self.model_params_cp = deepcopy(self.model_params)
            self.model_params_cp['n_steps'] = force_iter
            self.model = TabNetRegressor(**self.model_params_cp)
            self.model.fit(train['feature'].values, )
        else:
            self.model = TabNetRegressor(**self.model_params)
        
        if valid is not None:
            self.index = valid['label'].index

            self.model.fit(
                train['feature'].values, 
                train['label'].values,
                eval_name=['valid'],
                eval_set=[(valid['feature'].values, valid['label'].values)],
                eval_metric=[self.metric()],
                patience=self.ret_stop,
                **kwargs,
            )
        else:
            self.model.fit(train['feature'].values, train['label'].values, **kwargs)
        
        self.evals_result = {
            "valid": {
                "loss": self.model.history['loss'],
                "top_ret": self.model.history['valid_top_ret'],
                "lr": self.model.history['lr'],
            },
        }
        
        return self


class DoubleEnsembleModel(TopRetModelBase):
    """Double Ensemble Model"""

    def __init__(
        self,
        ret,
        top = 0.1,
        ret_stop = 3,
        num_models=100,
        stop_models=5,
        enable_sr=True,
        enable_fs=True,
        alpha1=1.0,
        alpha2=1.0,
        bins_sr=10,
        bins_fs=5,
        decay=0.9,
        sample_ratios=None,
        sub_weights=None,
        **kwargs
    ):
        self.ret = ret
        self.top = top
        self.ret_stop = ret_stop
        self.num_models = num_models  # the number of sub-models
        self.enable_sr = enable_sr
        self.enable_fs = enable_fs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.bins_sr = bins_sr
        self.bins_fs = bins_fs
        self.decay = decay
        self.stop_models = stop_models

        if sample_ratios is None: 
            sample_ratios = [0.8, 0.7, 0.6, 0.5, 0.4]
        if sub_weights is None:
            sub_weights = [1] * self.num_models
        if not len(sample_ratios) == bins_fs:
            raise ValueError("The length of sample_ratios should be equal to bins_fs.")
        self.sample_ratios = sample_ratios
        if not len(sub_weights) == num_models:
            raise ValueError("The length of sub_weights should be equal to num_models.")
        self.sub_weights = sub_weights
        self.model_params = {}
        self.model_params.update(kwargs)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame = None, force_iter: int = None, **kwargs):
        self.ensemble = []
        self.sub_features = []
        self.best_score = -np.inf
        self.best_iter = 0
        self.evals_result = None if valid is None else {"valid": {"top_ret": []}}
        if train.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = train["feature"], train["label"]
        # initialize the sample weights
        N, F = x_train.shape
        weights = pd.Series(np.ones(N, dtype=float))
        # initialize the features
        features = x_train.columns
        pred_sub = pd.DataFrame(np.zeros((N, self.num_models), dtype=float), index=x_train.index)
        if valid is not None:
            pred_valid_sub = pd.DataFrame(np.zeros((valid['feature'].shape[0], self.num_models), dtype=float), index=valid['feature'].index)
        # train sub-models
        for k in range(self.num_models if force_iter is None else force_iter):
            self.sub_features.append(features)
            print(f"[Model {k + 1}] Training sub-model: {k + 1}/{self.num_models}")
            model_k = self.train_submodel(train, valid, weights, features)
            self.ensemble.append(model_k)
            # no further sample re-weight and feature selection needed for the last sub-model
            if k + 1 == self.num_models:
                break
            print(f"[Model {k + 1}] Retrieving loss curve and loss values...")
            loss_curve = self.retrieve_loss_curve(model_k, train, features)
            pred_k = self.predict_sub(model_k, train, features)
            if valid is not None:
                pred_valid_k = self.predict_sub(model_k, valid, features)

            pred_sub.iloc[:, k] = pred_k
            if valid is not None:
                pred_valid_sub.iloc[:, k] = pred_valid_k
            
            pred_ensemble = (pred_sub.iloc[:, : k + 1] * self.sub_weights[0 : k + 1]).sum(axis=1) / np.sum(
                self.sub_weights[0 : k + 1]
            )
            if valid is not None:
                pred_valid_ensemble = (pred_valid_sub.iloc[:, :k + 1] * 
                    self.sub_weights[:k + 1]).sum(axis=1) / np.sum(self.sub_weights[:k + 1])

            loss_values = pd.Series(self.get_loss(y_train.values.squeeze(), pred_ensemble.values))

            if valid is not None:
                iter_score = self.ret.loc[pred_valid_ensemble.groupby(level=0).apply(lambda x: x.sort_values(
                    ascending=False).iloc[:int(len(x) * self.top)]).droplevel(0).index].groupby(level=0).mean().squeeze().sum()
                if iter_score > self.best_score:
                    self.best_score = iter_score
                    self.best_iter = k
                print(f"[Model {k + 1}] Evaluation Top Return: {iter_score:.4f}")
                self.evals_result['valid']['top_ret'].append(iter_score)
                if k - self.best_iter >= self.stop_models:
                    print(f"[Model Stop] Top Return didn't improve for {self.stop_models} models")
                    print(f"[Model Stop] Best Score is {self.best_score}, with {self.best_iter + 1} models")
                    break

            if self.enable_sr:
                print(f"[Model {k + 1}] Sample re-weighting...")
                weights = self.sample_reweight(loss_curve, loss_values, k + 1)

            if self.enable_fs:
                print(f"[Model {k + 1}] Feature selection...")
                features = self.feature_selection(train, loss_values)
        
        self.ensemble = self.ensemble[:self.best_iter + 1]

    def metric(self, y_true, y_pred):
        y_true = self.ret.loc[self.valid_index]
        y_pred = pd.Series(y_pred, index=y_true.index)
        select = y_pred.groupby(level=0).apply(
            lambda x: x.sort_values(ascending=False).iloc[:int(len(x) * self.top)]
        ).droplevel(0).index
        ret = y_true.squeeze().loc[select].groupby(level=0).mean().sum()
        return ("top_ret", ret, True)

    def early_stopping_for_ret(self, stop_round):
        cmp_op = [lt, gt]
        best_score = [np.inf, -np.inf]
        best_iter = [0, 0]
        best_score_list = [(), ()]
        
        def _callback(env):
            for i, evals in enumerate(env.evaluation_result_list):
                if cmp_op[i](evals[2], best_score[i]):
                    best_score[i] = evals[2]
                    best_iter[i] = env.iteration
                    best_score_list[i] = evals
                # we only consider whether the top return rises
                if evals[1] == 'top_ret':
                    if env.iteration - best_iter[i] >= stop_round:
                        print(f"[Early Stop]: Best top return: {best_score[i]}," 
                            f" best iteration: {best_iter[i]}")
                        raise EarlyStopException(best_iter[i], [best_score_list[i]])
                    
        return _callback 

    def train_submodel(self, train, valid, weights, features):
        model = lgb.LGBMRegressor(**self.model_params)
        if valid is not None:
            self.valid_index = valid.index
            eval_dict = dict(
                eval_names=['valid'],
                eval_set=[(valid['feature'].loc[:, features].values, valid['label'].squeeze())],
                eval_metric=self.metric,
                callbacks=[self.early_stopping_for_ret(self.ret_stop)],
            )
        else:
            eval_dict = {}
        model.fit(
            train['feature'].loc[:, features].values, 
            train['label'].squeeze().values,
            sample_weight=weights,
            verbose=False,
            **eval_dict,
        )
        return model

    def sample_reweight(self, loss_curve, loss_values, k_th):
        """
        the SR module of Double Ensemble
        -------------------------------

        loss_curve: the shape is NxT
            the loss curve for the previous sub-model, where the element (i, t) if the error on the i-th sample
            after the t-th iteration in the training of the previous sub-model.
        loss_values: the shape is N
            the loss of the current ensemble on the i-th sample.
        k_th: the index of the current sub-model, starting from 1
        return: weights
            the weights for all the samples.
        """
        # normalize loss_curve and loss_values with ranking
        loss_curve_norm = loss_curve.rank(axis=0, pct=True)
        loss_values_norm = (-loss_values).rank(pct=True)

        # calculate l_start and l_end from loss_curve
        N, T = loss_curve.shape
        part = np.maximum(int(T * 0.1), 1)
        l_start = loss_curve_norm.iloc[:, :part].mean(axis=1)
        l_end = loss_curve_norm.iloc[:, -part:].mean(axis=1)

        # calculate h-value for each sample
        h1 = loss_values_norm
        h2 = (l_end / l_start).rank(pct=True)
        h = pd.DataFrame({"h_value": self.alpha1 * h1 + self.alpha2 * h2})

        # calculate weights
        h["bins"] = pd.cut(h["h_value"], self.bins_sr)
        h_avg = h.groupby("bins")["h_value"].mean()
        weights = pd.Series(np.zeros(N, dtype=float))
        for b in h_avg.index:
            weights[h["bins"] == b] = 1.0 / (self.decay**k_th * h_avg[b] + 0.1)
        return weights

    def feature_selection(self, train, loss_values):
        """
        the FS module of Double Ensemble
        ---------------------------------

        df_train: the shape is NxF
        loss_values: the shape is N
            the loss of the current ensemble on the i-th sample.
        return: result feature, in the form of pandas.Index
        """
        x_train, y_train = train["feature"], train["label"]
        features = x_train.columns
        N, F = x_train.shape
        g = pd.DataFrame({"g_value": np.zeros(F, dtype=float)})
        M = len(self.ensemble)

        # shuffle specific columns and calculate g-value for each feature
        x_train_tmp = x_train.copy()
        for i_f, feat in enumerate(features):
            x_train_tmp.loc[:, feat] = np.random.permutation(x_train_tmp.loc[:, feat].values)
            pred = pd.Series(np.zeros(N), index=x_train_tmp.index)
            for i_s, submodel in enumerate(self.ensemble):
                pred += (
                    pd.Series(
                        submodel.predict(x_train_tmp.loc[:, self.sub_features[i_s]].values), 
                        index=x_train_tmp.index
                    ) / M
                )
            loss_feat = self.get_loss(y_train.values.squeeze(), pred.values)
            g.loc[i_f, "g_value"] = np.mean(loss_feat - loss_values) / (np.std(loss_feat - loss_values) + 1e-7)
            x_train_tmp.loc[:, feat] = x_train.loc[:, feat].copy()

        # one column in train features is all-nan # if g['g_value'].isna().any()
        g["g_value"].replace(np.nan, 0, inplace=True)

        # divide features into bins_fs bins
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)

        # randomly sample features from bins to construct the new features
        res_feat = []
        sorted_bins = sorted(g["bins"].unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
        return pd.Index(set(res_feat))

    def get_loss(self, label, pred):
        return (label - pred) ** 2

    def retrieve_loss_curve(self, model, train, features):
        num_trees = model.n_estimators
        x_train, y_train = train["feature"].loc[:, features], train["label"].squeeze()
        N = x_train.shape[0]
        loss_curve = pd.DataFrame(np.zeros((N, num_trees)))
        pred_tree = np.zeros(N, dtype=float)
        for i_tree in range(num_trees):
            pred_tree += model.predict(x_train.values, start_iteration=i_tree, num_iteration=1)
            loss_curve.iloc[:, i_tree] = self.get_loss(y_train.values, pred_tree)
        return loss_curve

    def predict(self, test: pd.DataFrame):
        if self.ensemble is None:
            raise ValueError("model is not fitted yet!")
        x_test = test['feature']
        pred = pd.Series(np.zeros(x_test.shape[0]), index=x_test.index)
        for i_sub, submodel in enumerate(self.ensemble):
            feat_sub = self.sub_features[i_sub]
            pred += (
                pd.Series(submodel.predict(x_test.loc[:, feat_sub].values), index=x_test.index)
                * self.sub_weights[i_sub]
            )
        pred = pred / np.sum(self.sub_weights)
        return pred

    def predict_sub(self, submodel, test, features):
        x_data = test["feature"].loc[:, features]
        pred_sub = pd.Series(submodel.predict(x_data.values), index=x_data.index)
        return pred_sub

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """get feature importance"""
        res = []
        for _model, _weight in zip(self.ensemble, self.sub_weights):
            res.append(pd.Series(_model.feature_importance(*args, **kwargs), index=_model.feature_name()) * _weight)
        return pd.concat(res, axis=1, sort=False).sum(axis=1).sort_values(ascending=False)


class FusionModel(TopRetModelBase):
    def __init__(
        self,
        ret,
        models: list,
        model_kwargs: list[dict],
        fusion: TopRetModelBase = None,
        fusion_kwargs: dict = {},
        top = 0.1,
        ret_stop = 10,
        method: str = 'stacking',
    ):
        self.ret = ret
        self.top = top
        self.ret_stop = ret_stop
        self.fusion = fusion
        self.fusion_class = fusion
        self.fusion_kwargs = fusion_kwargs
        assert method != 'average' and fusion is not None,\
            "If fusion method is not simple average, a second-level model should be specified"
        self.models = models
        self.model_class = models
        for kw in model_kwargs:
            kw.update({'ret': ret})
        self.model_kwargs = model_kwargs
        self.method = method

        method = getattr(self, f"_{self.method}_fit")
        if method is None:
            raise ValueError(f"Method {self.method} is not supported")
            
        assert len(self.models) == len(self.model_kwargs), \
            'The number of model and keyword arguments should match!'
        
        self.evals_result = None

    def _average_fit(self, train: pd.DataFrame, valid: pd.DataFrame, force_iter: int = None, **kwargs):
        models = [self.models[i](self.model_kwargs[i]) for i in range(len(self.models))]
        self.models = []
        for model in models:
            model.fit(train, valid, force_iter, **kwargs)
            self.models.append(model)

    def _stacking_fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        force_iter: int = None,
        kfold: int = None,
        **kwargs,
    ):
        models = [self.model_class[i](**self.model_kwargs[i]) for i in range(len(self.models))]
        self.fusion = self.fusion_class(**self.fusion_kwargs)
        if kfold is None:
            kfold = len(models)
        self.models = []
        kf = KFold(n_splits=kfold)
        pred_trains_ = []
        pred_valids_ = []
        for model in models:
            # for every model, we apply kfold train
            pred_trains = []
            pred_valids = []
            for train_index, test_index in kf.split(train):
                # test by 1 fold, train by the others
                model.fit(train.iloc[train_index], None, force_iter, **kwargs)
                # the test fold is used as the data source for fusion model
                pred_trains.append(model.predict(train.iloc[test_index]))
                if valid is not None:
                    pred_valids.append(model.predict(valid))
            pred_train = pd.concat(pred_trains, axis=0)
            if valid is not None:
                pred_valid = pd.concat(pred_valids, axis=1).mean(axis=1)
            self.models.append(model)
            pred_trains_.append(pred_train)
            if valid is not None:
                pred_valids_.append(pred_valid)
        pred_train = pd.concat(pred_trains_, axis=1)
        pred_train = pd.concat([pred_train, train['label']], axis=1, keys=['feature', 'label'])
        if valid is not None:
            pred_valid = pd.concat(pred_valids_, axis=1)
            pred_valid = pd.concat([pred_valid, valid['label']], axis=1, keys=['feature', 'label'])
            self.fusion.fit(pred_train, pred_valid, force_iter, **kwargs)
            self.evals_result = self.fusion.evals_result
        else:
            self.fusion.fit(pred_train, force_iter, **kwargs)

    def _voting_fit(self, train: pd.DataFrame, valid: pd.DataFrame, force_iter: int = None, **kwargs):
        raise NotImplementedError("Voting is still under development")
    
    def _average_predict(self, test: pd.DataFrame, model_weight: None):
        preds = []
        for model in self.models:
            preds.append(model.predict(test))
        pred = pd.concat(preds, axis=1)
        if model_weight is None:
            model_weight = [1 for _ in range(len(self.models))]
        pred = pred * model_weight
        pred = pred.sum(axis=1) / sum(model_weight)
        return pred
    
    def _stacking_predict(self, test: pd.DataFrame, kfold: int = None):
        pred_test = []
        for model in self.models:
            pred_test.append(model.predict(test))
        pred_test = pd.concat(pred_test, axis=1)
        pred_test = pd.concat([pred_test, test['label']], axis=1, keys=['feature', 'label'])
        return self.fusion.predict(pred_test)
    
    def _voting_predict(self, test: pd.DataFrame, model_weight: None):
        raise NotImplementedError("Voting is still under development")

    def fit(
        self, 
        train: pd.DataFrame,
        valid: pd.DataFrame = None,
        force_iter: int = None,
        **kwargs
    ):
        getattr(self, f'_{self.method}_fit')(train, valid, force_iter, **kwargs)
        return self
    
    def predict(self, test: pd.DataFrame, **kwargs):
        return getattr(self, f"_{self.method}_predict")(test, **kwargs)
        