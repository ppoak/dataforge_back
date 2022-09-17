import numpy as np
import pandas as pd


def compute_ic(
    pred_ret: pd.DataFrame,
    grouper: pd.Series = None,
    score_col: str = 'score',
    label_col: str = 'label',
    date_level: int = 0,
):
    # compute ic
    ic = pred_ret.groupby(level=date_level).corr()\
        .loc[(slice(None), label_col), score_col].droplevel(1 - date_level)
    ic = ic.sort_index()
    icmm = ic.groupby(lambda x: x.strftime('%Y-%m')).mean()
    icmm = icmm.sort_index()
    if grouper is not None:
        gic = pred_ret.groupby([pd.Grouper(level=date_level), grouper]).corr()\
            .loc[(slice(None), slice(None), label_col), score_col]
        gic = gic.sort_index()
    
    return {"ic": ic} if grouper is None else {
        "ic": ic,
        "gic": gic
    }
    
def rebalance(
    pred_ret: pd.DataFrame,
    benchmark: pd.Series = None,
    score_col: str = 'score',
    label_col: str = 'label',
    date_level: int = 0,
    weights: pd.Series = None,
    N: int = 5,
    period: int = 1,
    commission_ratio: float = 0.001,
):
    # only consider the dropna values
    pred_label_drop = pred_ret.dropna(subset=[score_col])
    quantiles = pred_label_drop.groupby(level=date_level)[score_col].apply(
        lambda x: pd.Series([i for i in range(N, 1, -1) for _ in range(len(x) // N)] + 
            [1] * (len(x) - (len(x) // N) * (N - 1)), index=x.sort_values(ascending=False).index.get_level_values(1))
    )

    # constructing weight and compute profit without commission
    if weights is None:
        weights = pd.Series(np.ones_like(quantiles), index=quantiles.index)
        weights = weights.groupby([quantiles, pd.Grouper(level=date_level)]).apply(lambda x: x / x.sum() * 1 / period)
    profit = weights.groupby([quantiles, pd.Grouper(level=0)]).apply(
        lambda x: (x.droplevel(0) * pred_ret.loc[x.index.get_level_values(0)[0], label_col]).sum()
    ).unstack(level=0)
    
    panel_weights = weights.unstack().fillna(0).stack()
    turnovers = []
    for i in range(period):
        # daily rebalance isn't th same portfolio, only with the interval
        # period + 1 can be considered in the same portfolio
        subport_date = panel_weights.index.levels[0][i::period]
        turnover = panel_weights.loc[subport_date].groupby(quantiles).apply(
            lambda x: ((x.unstack().fillna(0) - 
                x.unstack().fillna(0).shift(1)).fillna(0).stack())\
                # if y[y>0].abs().sum(), computing the long turnover
                # if y[y<0].abs().sum(), computing the short turnover
                # if y.abs().sum(), computing both turnover
                .groupby(level=date_level).apply(lambda y: y[y >= 0].abs().sum())
        )
        turnover.iloc[0] = 1 / period
        turnovers.append(turnover)
    turnover = pd.concat(turnovers, axis=0).sort_index().unstack(level=0)
    profit -= commission_ratio * turnover
    profit = profit.shift(period + 1).fillna(0)
    benchmark.iloc[:(period + 1), :] = 0
    profit['benchmark'] = benchmark

    exret = profit.iloc[:, :-1].subtract(profit.iloc[:, -1], axis=0)
    exmmd = ((exret + 1).cumprod() - (exret + 1).cumprod().cummax()) / (exret + 1).cumprod().cummax()
    mmd = ((profit + 1).cumprod() - (profit + 1).cumprod().cummax()) / (profit + 1).cumprod().cummax()
    
    return {
        "profit": profit,
        "exprofit": exret,
        "mmd": mmd,
        "exmmd": exmmd,
        "turnover": turnover,
    }
        