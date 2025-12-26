# region imports
from AlgorithmImports import *
# endregion

class CloneofCryptoStatic(QCAlgorithm):

    def initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        self.symbols = ["ETHUSD", "SOLUSD", "BTCUSD"]
        for symbol in self.symbols:
            self.AddCryptoAsset(symbol, Resolution.Hour)

        self.SetBrokerageModel(BrokerageName.Coinbase, AccountType.Margin)

        # Rolling windows for price data
        self.prices = {}
        self.returns = {}

        for symbol in self.symbols:
            self.prices[symbol] = RollingWindow[float](252)
            self.returns[symbol] = RollingWindow[float](252)

        self.lag_window = 30  # minutes for lag calculation
        self.last_lag_calc = None

    def OnData(self, data):
        if not data.HasData:
            return

        # Update rolling windows
        for symbol in self.symbols:
            if symbol in data:
                price = data[symbol].Close
                self.prices[symbol].Add(price)

                if self.prices[symbol].Count > 1:
                    ret = (price - self.prices[symbol][1]) / self.prices[symbol][1]
                    self.returns[symbol].Add(ret)

        # Check if we have enough data
        if not all(self.prices[s].IsReady for s in self.symbols):
            return

        # Calculate optimal lag
        if self.last_lag_calc is None or \
           (self.Time - self.last_lag_calc).total_seconds() > 3600:
            self._calcular_lag_optimo()
            self._detectar_señal_lagged()
            self.last_lag_calc = self.Time

    def _calcular_lag_optimo(self):
        """Calculate optimal lag between crypto assets"""
        if not all(self.returns[s].Count >= 60 for s in self.symbols):
            return

        # Simple cross-correlation implementation
        eth_ret = [self.returns["ETHUSD"][i] for i in range(min(60, self.returns["ETHUSD"].Count))]
        btc_ret = [self.returns["BTCUSD"][i] for i in range(min(60, self.returns["BTCUSD"].Count))]

        max_corr = 0
        best_lag = 0

        for lag in range(-self.lag_window, self.lag_window + 1):
            if lag < 0:
                corr = self._calculate_correlation(eth_ret[-lag:], btc_ret[:lag])
            elif lag > 0:
                corr = self._calculate_correlation(eth_ret[:-lag], btc_ret[lag:])
            else:
                corr = self._calculate_correlation(eth_ret, btc_ret)

            if abs(corr) > abs(max_corr):
                max_corr = corr
                best_lag = lag

        self.best_lag = best_lag
        self.Debug(f"Optimal lag: {best_lag}, Correlation: {max_corr:.4f}")

    def _calculate_correlation(self, x, y):
        """Calculate Pearson correlation"""
        if len(x) < 2 or len(y) < 2:
            return 0

        x = list(x)
        y = list(y)

        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denom_x = sum((x[i] - mean_x) ** 2 for i in range(len(x))) ** 0.5
        denom_y = sum((y[i] - mean_y) ** 2 for i in range(len(y))) ** 0.5

        if denom_x == 0 or denom_y == 0:
            return 0

        return numerator / (denom_x * denom_y)

    def _detectar_señal_lagged(self):
        """Detect trading signals based on lagged relationships"""
        # Simplified signal generation
        self.Debug(f"Detecting signals with lag: {self.best_lag}")
