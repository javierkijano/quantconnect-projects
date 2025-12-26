# region imports
from AlgorithmImports import *
# endregion

class MeanReversionLag(QCAlgorithm):
    """Lead-lag strategy for SPY with dynamic lag detection"""

    def initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.AddEquity("SPY", Resolution.Minute)
        self.AddEquity("IWM", Resolution.Minute)
        
        self.symbol_lead = "SPY"
        self.symbol_lag = "IWM"
        
        # Windows for lead-lag detection
        self.lead_prices = RollingWindow[float](390)  # Full trading day
        self.lag_prices = RollingWindow[float](390)
        
        self.lead_returns = RollingWindow[float](60)  # 1-hour window
        self.lag_returns = RollingWindow[float](60)
        
        self.optimal_lag = 0
        self.last_lag_calc = None
        self.lag_calc_interval = 60  # minutes

    def OnData(self, data):
        if not data.HasData:
            return

        # Update price windows
        if self.symbol_lead in data and data[self.symbol_lead].Volume > 0:
            lead_price = data[self.symbol_lead].Close
            self.lead_prices.Add(lead_price)
            
            if self.lead_prices.Count > 1:
                lead_ret = (lead_price - self.lead_prices[1]) / self.lead_prices[1]
                self.lead_returns.Add(lead_ret)

        if self.symbol_lag in data and data[self.symbol_lag].Volume > 0:
            lag_price = data[self.symbol_lag].Close
            self.lag_prices.Add(lag_price)
            
            if self.lag_prices.Count > 1:
                lag_ret = (lag_price - self.lag_prices[1]) / self.lag_prices[1]
                self.lag_returns.Add(lag_ret)

        # Recalculate lag periodically
        if self.last_lag_calc is None or \
           (self.Time - self.last_lag_calc).total_seconds() > (self.lag_calc_interval * 60):
            self._calculate_optimal_lag()
            self.last_lag_calc = self.Time

        # Generate trading signal
        if self.lead_returns.IsReady and self.lag_returns.IsReady:
            self._generate_signal()

    def _calculate_optimal_lag(self):
        """Calculate optimal lag between lead and lag assets"""
        if not self.lead_returns.IsReady or not self.lag_returns.IsReady:
            return

        lead_ret_list = [self.lead_returns[i] for i in range(self.lead_returns.Count)]
        lag_ret_list = [self.lag_returns[i] for i in range(self.lag_returns.Count)]

        max_corr = 0
        best_lag = 0
        max_lag_range = 30  # minutes

        for lag in range(-max_lag_range, max_lag_range + 1):
            if lag < 0:
                # Lead lags behind
                corr = self._correlation(lead_ret_list[-lag:], lag_ret_list[:lag])
            elif lag > 0:
                # Lag leads
                corr = self._correlation(lead_ret_list[:-lag], lag_ret_list[lag:])
            else:
                corr = self._correlation(lead_ret_list, lag_ret_list)

            if abs(corr) > abs(max_corr):
                max_corr = corr
                best_lag = lag

        self.optimal_lag = best_lag
        self.Debug(f"Optimal lag: {best_lag} minutes, Correlation: {max_corr:.4f}")

    def _correlation(self, x, y):
        """Calculate Pearson correlation coefficient"""
        if len(x) < 2 or len(y) < 2:
            return 0.0

        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denom_x = (sum((x[i] - mean_x) ** 2 for i in range(len(x))) ** 0.5)
        denom_y = (sum((y[i] - mean_y) ** 2 for i in range(len(y))) ** 0.5)

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return num / (denom_x * denom_y)

    def _generate_signal(self):
        """Generate trading signal based on lead-lag relationship"""
        if self.lead_returns.Count == 0:
            return

        # Z-score of lead returns
        lead_ret_list = [self.lead_returns[i] for i in range(min(20, self.lead_returns.Count))]
        if len(lead_ret_list) >= 2:
            mean_lead = sum(lead_ret_list) / len(lead_ret_list)
            std_lead = (sum((r - mean_lead) ** 2 for r in lead_ret_list) / len(lead_ret_list)) ** 0.5

            if std_lead > 0:
                latest_lead_ret = self.lead_returns[0]
                z_score = (latest_lead_ret - mean_lead) / std_lead

                # Simple trading rule based on z-score
                if z_score > 2:
                    self._place_trade(1)
                elif z_score < -2:
                    self._place_trade(-1)
                else:
                    self._place_trade(0)

    def _place_trade(self, signal):
        """Place trade based on signal"""
        if signal == 0:
            return

        qty = int(self.Portfolio.Cash / self.Securities[self.symbol_lag].Price * 0.25)
        if qty > 0:
            if signal > 0 and not self.Portfolio[self.symbol_lag].IsLong:
                self.Buy(self.symbol_lag, qty)
            elif signal < 0 and self.Portfolio[self.symbol_lag].IsLong:
                self.Liquidate(self.symbol_lag)
