import numpy as np
import pandas as pd
from tools import parse_date


class BackTest:

    def __init__(
        self,
        ohlcv: pd.DataFrame,
    ) -> None:
        self.ohlcv = ohlcv
    
    def __str__(self) -> str:
        return f"BackTest on {self.ohlcv}"

    def __repr__(self) -> str:
        return self.__str__()

    def run(
        self,
        start: str,
        end: str,
        weight: pd.DataFrame = None,
        cash: float = 1e4,
        minimum_trade: int = 100,
        commission: float = 5e-3,
        bilateral_commission: bool = True,
        trade_on: str = 'close',
    ):
        start = parse_date(start)
        end = parse_date(end)
        
        if weight is None:
            weight = pd.DataFrame(index=self.ohlcv['close'].index, columns=self.ohlcv['close'].columns)
        else:
            weight = weight.reindex(index=self.ohlcv['close'].index, columns=self.ohlcv['close'].columns)
        weight = weight.values
        
        value = np.full(weight.shape[0], fill_value=np.nan)
        value[0] = cash
        cash = np.full(weight.shape[0], fill_value=np.nan)
        cash[0] = value[0]
        share = np.zeros_like(weight)
                
        for d in range(1, weight.shape[0]):
            w, p = self.generate_weight(share[:d], weight[:d], cash[:d], value[:d])

            ap = p <= self.ohlcv.high[d] and p >= self.ohlcv.low[d]
            ap = ap | np.isnan(p)
            p = np.isnan(p) * getattr(self.ohlcv, trade_on)[d] + (~np.isnan(p) & ap) * np.nan_to_num(p)
            if len(ap[ap == False]) > 0:
                print("The following asset failed in trading:")
                print(ap[ap == False])
            w[~ap] = np.nan

            sd = w * value[d - 1] / p // minimum_trade * minimum_trade
            share[d] = np.isnan(sd) * share[d - 1] + ~np.isnan(sd) * np.nan_to_num(sd)
            delta_share = share[d] - share[d - 1]
            cash[d] = (cash[d - 1] - delta_share[delta_share >= 0] @ p[delta_share >= 0] * (1 + commission)
                 - delta_share[delta_share < 0] @ p[delta_share < 0] * (1 + commission * bilateral_commission))
            value[d] = cash[d] + share[d] * p
        
        self.weight = pd.DataFrame(weight, 
            index=self.ohlcv['close'].index, 
            columns=self.ohlcv['close'].columns
        )
        self.share = pd.DataFrame(share, 
            index=self.ohlcv['close'].index, 
            columns=self.ohlcv['close'].columns
        )
        self.cash = pd.Series(cash, 
            index=self.ohlcv['close'].index, 
        )
        self.value = pd.Series(value, 
            index=self.ohlcv['close'].index, 
        )
        return self

    def evaluate(self):
        pass

    def generate_weight(
        self,
        share: np.ndarray, 
        weight: np.ndarray,
        cash: np.ndarray, 
        value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (weight[-1, :], np.full(weight.shape[1], fill_value=np.nan))

