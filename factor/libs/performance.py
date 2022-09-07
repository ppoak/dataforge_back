import numpy as np
import pandas as pd


def rebalance(
    pred_label: pd.DataFrame,
    benchmark: pd.Series = None,
    score_col: str = 'score',
    label_col: str = 'label',
    date_level: int = 0,
    N: int = 5,
    period: int = 1,
    reverse: bool = False,
    commission_ratio: float = 0.03,
):
    if reverse:
        pred_label[score_col] *= -1

    # Group1 ~ Group5 only consider the dropna values
    pred_label_drop = pred_label.dropna(subset=[score_col])
    # Group
    quantiles = pred_label_drop.groupby(level=date_level)[score_col].apply(
        lambda x: pd.Series([i for i in range(1, N) for _ in range(len(x) // N)] + 
            [N] * (len(x) - (len(x) // N) * (N - 1)), index=x.sort_values().index.get_level_values(1))
    )
    quantiles = quantiles.sort_index()

    # constructing weight and compute profit without commission
    weights = pd.Series(np.ones_like(quantiles), index=quantiles.index)
    weights = weights.groupby([quantiles, pd.Grouper(level=date_level)]).apply(lambda x: x / x.sum() * 1 / (period + 1) )
    profit = weights.groupby([quantiles, pd.Grouper(level=0)]).apply(
        lambda x: (x.droplevel(0) * pred_label.loc[x.index.get_level_values(0)[0], label_col]).sum()
    ).unstack(level=0)
    
    panel_weights = weights.unstack().stack(dropna=False).fillna(0)
    turnovers = []
    for i in range(period + 1):
        # daily rebalance isn't th same portfolio, only with the interval
        # period + 1 can be considered in the same portfolio
        subport_date = panel_weights.index.levels[0][i::(period + 1)]
        turnover = panel_weights.loc[subport_date].groupby(quantiles).apply(
            lambda x: (x - x.groupby(level=1-date_level).shift(1).fillna(0))\
                # if y[y>0].abs().sum(), computing the long turnover
                # if y[y<0].abs().sum(), computing the short turnover
                # if y.abs().sum(), computing both turnover
                .groupby(level=date_level).apply(lambda y: y.abs().sum()
            )
        )
        turnovers.append(turnover)
    turnover = pd.concat(turnovers, axis=0).sort_index().unstack(level=0)
    profit -= commission_ratio * turnover
    profit = profit.shift(1).fillna(0)
    profit['benchmark'] = benchmark
    return {
        "profit": profit,
        "turnover": turnover,
    }
        