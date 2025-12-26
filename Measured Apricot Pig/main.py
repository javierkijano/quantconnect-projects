# region imports
from AlgorithmImports import *
# endregion

class MeasuredApricotPig(QCAlgorithm):
    """Simple 3-way diversified portfolio"""

    def initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # Add assets
        self.AddEquity("SPY", Resolution.Daily)  # Stocks
        self.AddEquity("BND", Resolution.Daily)  # Bonds
        self.AddEquity("AAPL", Resolution.Daily)  # Tech

        self.symbols = ["SPY", "BND", "AAPL"]
        self.weights = {s: 1.0 / len(self.symbols) for s in self.symbols}

        self.initialized = False

    def OnData(self, data):
        # Initialize portfolio on first call
        if not self.initialized:
            self._initialize_portfolio()
            self.initialized = True
            return

    def _initialize_portfolio(self):
        """Initialize equal-weight portfolio"""
        total_value = self.Portfolio.TotalPortfolioValue

        for symbol in self.symbols:
            if symbol in self.Securities:
                price = self.Securities[symbol].Price
                if price > 0:
                    target_value = total_value * self.weights[symbol]
                    qty = int(target_value / price)
                    
                    if qty > 0:
                        self.Buy(symbol, qty)
                        self.Debug(f"Bought {qty} shares of {symbol} at ${price:.2f}")

        self.Debug("Portfolio initialized with equal weights: SPY 33%, BND 33%, AAPL 33%")

    def OnEndOfDay(self):
        """Log portfolio value"""
        self.Plot("Portfolio", "Value", self.Portfolio.TotalPortfolioValue)
