import pandas as pd
from libs.models import TopRetModelBase

class SomeModel(TopRetModelBase):

    def __init__(self, ret, top, ret_stop):
        # There are three main element in Model class definition:
        #   1. ret: return Series
        #   2. top: the top ratio for the stock selection
        #   3. ret_stop: the maximum epoches for top return not increasing, if exceed, exit.
        # The rest of the parameters are the model relative parameters
        super().__init__(ret, top, ret_stop)
    
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame = None, **kwargs) -> 'TopRetModelBase':
        # The required parameter is `train`, it is used for training, valid can be none. 
        # Once set to none, no evaluation method will be applied, and no early stop 
        # judgment will be applied.
        # NOTE: every model in the libs.model module got a `force_iter` parameter
        # which is for the force stop iteration, if epoch (or tree in tree model) is set,
        # while the `force_iter` is also applied, `force_iter` will be given the priority
        return super().fit(train, valid, **kwargs)
    
    def predict(self, test: pd.DataFrame) -> pd.Series:
        # Predict method should return a Series with multi-index labeling every stock
        # predict value on each single day.
        return super().predict(test)
    