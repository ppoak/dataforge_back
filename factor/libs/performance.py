import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots

def compute_ic(
    pred_label: pd.DataFrame,
    grouper: pd.Series = None,
    score_col: str = 'score',
    label_col: str = 'label',
    date_level: int = 0,
):
    # compute ic
    ic = pred_label.groupby(level=date_level).corr()\
        .loc[(slice(None), label_col), score_col].droplevel(1 - date_level)
    ic = ic.sort_index()
    icdm = ic.rolling(10).mean()
    icmm = ic.groupby(lambda x: x.strftime('%Y-%m')).mean()
    icmm = icmm.sort_index()
    if grouper is not None:
        gic = pred_label.groupby([pd.Grouper(level=date_level), grouper]).corr()\
            .loc[(slice(None), slice(None), label_col), score_col]
        gic = gic.sort_index()
        gicm = gic.rolling(10).mean()

    # plot the daily ic
    fig = subplots.make_subplots(rows=2 + (1 if grouper is not None else 0), cols=1)
    dailybar = go.Bar(x=ic.index, y=ic.values, name='Daily IC')
    dailyline = go.Scatter(x=icdm.index, y=icdm.values, 
        name='Daily IC Mean(10d)', mode='lines + markers')
    monthbar = go.Bar(x=icmm.index, y=icmm.values, name='Monthly IC')
    fig.add_trace(dailybar, row=1, col=1)
    fig.add_trace(dailyline, row=1, col=1)
    fig.add_trace(monthbar, row=2, col=1)
    if grouper is not None:
        gicdaily = go.Bar(x=gic.index, y=gic.values, name='Grouped IC')
        gicmean = go.Scatter(x=gicm.index, y=gicm.values, 
            name='Group IC Mean(10d)', mode='lines + markers')
        fig.add_trace(gicdaily, row=3, col=1)
        fig.add_trace(gicmean, row=3, col=1)

    fig['layout'].update(height=800 * (2 + (1 if grouper is not None else 0)))
    fig.show()
    
    return {"ic": ic} if grouper is None else {
        "ic": ic,
        "gic": gic
    }
    
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

    # plot the rebalance result
    fig = subplots.make_subplots(rows=4, cols=1)
    for group in profit.columns:
        fig.add_trace(go.Bar(
            x=profit.index, 
            y=profit[group].values, 
            name=f'{group} profit', 
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=profit.index, 
            y=(profit[group] + 1).cumprod(), 
            name=f'{group} equity', 
            mode='lines + markers',
        ), row=1, col=1)

    exret = profit.iloc[:, :-1].subtract(profit.iloc[:, -1], axis=0)
    for group in exret.columns:
        fig.add_trace(go.Bar(
            x=exret.index, 
            y=exret[group].values, 
            name=f'{group} excess profit', 
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=exret.index, 
            y=(exret[group] + 1).cumprod(), 
            name=f'{group} excess equity', 
            mode='lines + markers',
        ), row=2, col=1)
    
    mmd = ((exret + 1) - (exret + 1).cummax())
    for group in mmd.columns:
        fig.add_trace(go.Scatter(
            x=mmd.index, 
            y=mmd[group].values, 
            mode='lines + markers',
            name=f'{group} max draw down', 
        ), row=3, col=1)
    
    for group in turnover.columns:
        fig.add_trace(go.Scatter(
            x=turnover.index, 
            y=turnover[group].values, 
            mode='lines + markers',
            name=f'{group} turnover', 
        ), row=4, col=1)
    
    fig['layout'].update(height=800 * 4)
    fig.show()
    
    return {
        "profit": profit,
        "turnover": turnover,
    }
        