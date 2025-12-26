from AlgorithmImports import *


class SPYBuyAndHold(QCAlgorithm):
    """Simple SPY buy-and-hold benchmark for comparison"""
    
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(100000)
        
        self._spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self._invested = False
        
        self.debug("SPY Buy-and-Hold Benchmark initialized")
    
    def on_data(self, data: Slice):
        # Buy once and hold forever
        if not self._invested:
            self.set_holdings(self._spy, 1.0)
            self._invested = True

