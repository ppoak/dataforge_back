import datetime
import backtrader as bt


class Strategy(bt.Strategy):

    def log(self, text: str, datetime: datetime.datetime = None, hint: str = 'INFO'):
        datetime = datetime or self.data.datetime.date(0)
        print(f'[{hint}] {datetime}: {text}')

    def notify_order(self, order: bt.Order):
        if order.status in [order.Submitted, order.Accepted, order.Created]:
            return

        elif order.status in [order.Completed]:
            self.log(f'Trade <{order.executed.size}> <{order.info.get("name", "data")}> at <{order.executed.price:.2f}>')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log('Order canceled, margin, rejected or expired', hint='WARN')

        self.order = None