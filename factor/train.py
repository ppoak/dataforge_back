import numpy as np
import pandas as pd
from libs import (
    DNNModel,
    LGBModel,
    FusionModel, 
    LinearModel,
    RollingTrain
)


# data preparation
data = pd.read_parquet('data/intermediate/feature_info/normalized_dataset.parquet')
ret = pd.read_parquet('data/intermediate/forward_return/1d_open_open.parquet')

# for test, slice the foremost sample
train = data.loc['2018-01-01':'2018-06-30']
test = data.loc['2018-07-01':'2018-07-31']

# initialize the model
model = DNNModel(
    ret, 
    top=0.1,
    ret_stop=10,
    dnn_kwargs={
        "input_dim": 85,
        "output_dim": 1,
        "layers": (512,),
        "act": "LeakyReLU",
        "act_kwargs": {"negative_slope": 0.1, "inplace": False},
    },
    optimizer="AdamW",
    optimizer_kwargs={"lr": 0.001, "weight_decay": 0.001},
)
# model = LGBModel(ret, learning_rate=0.01)
# model = FusionModel(
#     ret, 
#     models=[LGBModel, DNNModel],
#     model_kwargs=[
#         {
#             "ret": ret,
#             "learning_rate": 0.01
#         },
#         {
#             "ret": ret, 
#             "ret_stop": 10, 
#             "dnn_kwargs": {
#                 "input_dim": 85,
#                 "output_dim": 1,
#                 "layers": (512,),
#                 "act": "LeakyReLU",
#                 "act_kwargs": {
#                     "negative_slope": 0.1, 
#                     "inplace": False
#                 },
#             },
#             "epoch": 130,
#         },
#     ],
#     fusion = LinearModel,
#     fusion_kwargs = {
#         "ret": ret, 
#         "in_feature": 2, 
#         "out_feature": 1, 
#         "epoch": 100
#     },
#     method='stacking',
# )

# single model training
model.fit(train, test)

# rolling model training
# roller = RollingTrain(
#     min_days=60, 
#     max_days=90,
#     pred_days=1,
#     learn_days=100,
#     exp_path='data/intermediate/results',
#     exp_name='dnn',
# )
# roller.rolling(model, data)
