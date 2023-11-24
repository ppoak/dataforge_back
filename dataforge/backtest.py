import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from pathlib import Path
from tools import parse_date


class Strategy(bt.Strategy):

    def log(self, text: str, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        print(f'[{datetime}]: {text}')

    def notify_order(self, order: bt.Order):
        """order notification"""
        # order possible status:
        # 'Created'、'Submitted'、'Accepted'、'Partial'、'Completed'、
        # 'Canceled'、'Expired'、'Margin'、'Rejected'
        # broker submitted or accepted order do nothing
        if order.status in [order.Submitted, order.Accepted, order.Created]:
            return

        # broker completed order, just hint
        elif order.status in [order.Completed]:
            self.log(f'Order <{order.executed.size}> <{order.info.get("name", "data")}> at <{order.executed.price:.2f}>')
            # record current bar number
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log('Order canceled, margin, rejected or expired')

        # except the submitted, accepted, and created status,
        # other order status should reset order variable
        self.order = None

    def notify_trade(self, trade):
        """trade notification"""
        if not trade.isclosed:
            # trade not closed, skip
            return
        # else, log it
        self.log(f'Gross Profit: {trade.pnl:.2f}, Net Profit {trade.pnlcomm:.2f}')


class Indicator(bt.Indicator):
    
    def log(self, text: str, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        print(f'[{datetime}]: {text}')    


class Analyzer(bt.Analyzer):

    def log(self, text: str, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        print(f'[{datetime}]: {text}')


class Observer(bt.Observer):

    def log(self, text: str, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        print(f'[{datetime}]: {text}')


class OrderTable(Analyzer):

    def __init__(self):
        self.orders = []

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.orders.append([
                    self.data.datetime.date(0),
                    order.info.get('name', 'data'), order.executed.size, 
                    order.executed.price, 'BUY']
                )
            elif order.issell():
                self.orders.append([
                    self.data.datetime.date(0),
                    order.info.get('name', 'data'), order.executed.size, 
                    order.executed.price, 'SELL']
                )
        
    def get_analysis(self):
        self.rets = pd.DataFrame(self.orders, columns=['datetime', 'asset', 'size', 'price', 'direction'])
        self.rets = self.rets.set_index('datetime')
        return self.orders


class Relocator:

    def _format(self, data: pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            return data
        elif isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
            return data.stack()
        else:
            raise ValueError("Malformed format of data")

    def __init__(
        self,
        price: pd.DataFrame,
        code_index: str = 'order_book_id',
        date_index: str = 'date_index',
        buy_column: str = "open",
        sell_column: str = "close",
        commision: float = 0.005,
    ):
        self.price = self._format(price)
        self.buy_price = self.price[buy_column] if isinstance(self.price, pd.DataFrame) else self.price
        self.sell_price = self.price[sell_column] if isinstance(self.price, pd.DataFrame) else self.price
        self.code_index = code_index
        self.date_index = date_index

    def turnover(
        self, 
        weight: pd.DataFrame | pd.Series, 
        side: str = 'both'
    ):
        weight = weight.reindex(pd.MultiIndex.from_product([
            self.weight.index.get_level_values(self.code_index).unique(), 
            self.weight.index.get_level_values(self.date_index).unique()
        ], names = [self.code_index, self.date_index])).fillna(0)

        preweight = weight.groupby(level=self.code_index).shift(1).fillna(0)
        delta = weight - preweight
        if side == 'both':
            return delta.groupby(level=self.date_index).apply(lambda x: x.abs().sum())
        elif side == 'buy':
            return delta.groupby(level=self.date_index).apply(lambda x: x[x > 0].abs().sum())
        elif side == 'sell':
            return delta.groupby(level=self.date_index).apply(lambda x: x[x < 0].abs().sum())
    
    def profit(
        self, 
        weight: pd.DataFrame | pd.Series, 
    ):
        weight = self._format(weight)
        comission = self.turnover(weight) * self.commision
        buy_price = self.buy_price.loc[weight.index].grouby(level=self.code_index).shift(1)
        sell_price = self.sell_price.loc[weight.index]
        ret = (sell_price - buy_price) / buy_price
        return weight.groupby(level=date_index).apply(lambda x: 
            (ret.loc[x.index] * x).sum() - commision.loc[x.index])

    def netvalue(
        self,
        weight: pd.DataFrame | pd.Series,    
    ):
        weight = self._format(weight)
        relocate_date = weight.index.get_level_values(self.date_index).unique()
        datetime_index = self.price.index.get_level_values(self.date_index).unique()
        lrd = relocate_date[0]
        lnet = (self.price.loc[d] * self.weight.loc[lrd]).sum()
        lcnet = 1
        net = pd.Series(np.ones(datetime_index.size), index=datetime_index)
        for d in datetime_index[1:]:
            cnet = (price.loc[d] * self.weight.loc[lrd]).sum() / lnet * lcnet
            lrd = relocate_date[relocate_date <= d][-1]
            if d == lrd:
                lcnet = cnet
                lnet = (price.loc[d] * self.weight.loc[lrd]).sum()
            net.loc[d] = cnet
        return net


class BackTrader:

    def __init__(
        self, 
        data: pd.DataFrame, 
        code_index: str = 'order_book_id',
        date_index: str = 'date_index',
    ):
        self.data = data
        self.data = self._valid(data)
        self.code_index = code_index
        self.date_index = date_index

    def _valid(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame) and not 'close' in data.columns:
            raise ValueError('Your data should at least have a column named close')
        
        required_col = ['open', 'high', 'low']
        if isinstance(data, pd.DataFrame):
            # you should at least have a column named close
            for col in required_col:
                if not col in data.columns and col != 'volume':
                    data[col] = data['close']
            if not 'volume' in data.columns:
                data['volume'] = 0
        else:
            # just a series, all ohlc data will be the same, volume set to 0
            data = data.to_frame(name='close')
            for col in required_col:
                data[col] = col
            data['volume'] = 0

        return data

    def run(
        self, 
        strategy: bt.Strategy, 
        start: str = None,
        stop: str = None,
        cash: float = 1e6,
        indicators: 'bt.Indicator | list' = None,
        analyzers: 'bt.Analyzer | list' = None,
        observers: 'bt.Observer | list' = None,
        coc: bool = False,
        verbose: bool = False,
        detail_img: str | Path = None,
        simple_img: str | Path = None,
        data_path: str | Path = None,
        **kwargs
    ):
        start = start or self.data.index.get_level_values(self.date_index).min()
        stop = stop or self.data.index.get_level_values(self.date_index).max()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(cash)
        if coc:
            cerebro.broker.set_coc(True)

        indicators = [indicators] if not isinstance(indicators, list) else indicators
        analyzers = [bt.analyzers.SharpeRatio, bt.analyzers.TimeDrawDown, bt.analyzers.TimeReturn, OrderTable]\
            if analyzers is None else [analyzers] if not isinstance(analyzers, list) else analyzers
        observers = [bt.observers.DrawDown] if observers is None else [observers]\
            if not isinstance(observers, list) else observers

        more = set(self.data.columns.to_list()) - set(['open', 'high', 'low', 'close', 'volume'])

        class _PandasData(bt.feeds.PandasData):
            lines = tuple(more)
            params = tuple(zip(more, [-1] * len(more)))
            
        # add data
        if isinstance(self.data.index, pd.MultiIndex):
            datanames = self.data.index.get_level_values(self.code_index).unique().to_list()
        else:
            datanames = ['data']
        for dn in datanames:
            d = self.data.xs(dn, level=self.code_index) if \
                isinstance(self.data.index, pd.MultiIndex) else self.data
            feed = _PandasData(dataname=d, fromdate=start, todate=stop)
            cerebro.adddata(feed, name=dn)
        
        if indicators is not None:
            for indicator in indicators:
                cerebro.addindicator(indicator)
        if strategy is not None:
            cerebro.addstrategy(strategy, **kwargs)
        for analyzer in analyzers:
            cerebro.addanalyzer(analyzer)
        for observer in observers:
            cerebro.addobserver(observer)
        
        result = cerebro.run()[0]

        timereturn = pd.Series(result.analyzers.timereturn.rets)

        if verbose:
            print('-' * 15 + "Sharpe" + '-' * 15)
            print(dict(result.analyzers.sharperatio.rets))
            print('-' * 15 + "Time Drawdown" + '-' * 15)
            print(dict(result.analyzers.timedrawdown.rets))
            print('-' * 15 + "Time Return" + '-' * 15)
            print(timereturn)
            print('-' * 15 + "Cummulative Return" + '-' * 15)
            print((timereturn + 1).cumprod())
            print('-' * 15 + "Order Table" + '-' * 15)
            print(result.analyzers.ordertable.rets)
        
        if detail_img is not None:
            if len(datanames) > 3:
                print(f"There are {len(datanames)} stocks, the image "
                      "may be nested and takes a long time to draw")
            figs = cerebro.plot(style='candel')
            fig = figs[0][0]
            fig.set_size_inches(18, 3 + 6 * len(datanames))
            fig.savefig(detail_img, dpi=300)

        if simple_img is not None:
            (timereturn + 1).cumprod().plot()
            plt.savefig(simple_img)
            
        if data_path is not None:
            with pd.ExcelWriter(data_path) as writer:
                timereturn.to_excel(writer, sheet_name='TimeReturn')
                (timereturn + 1).cumprod().to_excel(writer, sheet_name='CummulativeReturn')
                result.analyzers.ordertable.rets.to_excel(writer, sheet_name='OrderTable')
    
