# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class ResidualStateEngine(QCAlgorithm):
    """PCA residual-based portfolio optimization using energy framework"""

    def initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        self.symbols = ["SPY", "QQQ", "IWM", "VGK", "VXUS"]
        for symbol in self.symbols:
            self.AddEquity(symbol, Resolution.Daily)

        self.lookback = 252
        self.rebalance_days = 7
        self.days_since_rebalance = 0

        # Price windows
        self.prices = {}
        for symbol in self.symbols:
            self.prices[symbol] = RollingWindow[float](self.lookback)

        # Portfolio parameters
        self.k_factors = 3
        self.lambda_l2 = 0.05
        self.lambda_cov = 0.2
        self.trailing_stop_pct = 0.10

        self.initialized = False
        self.current_weights = {s: 0 for s in self.symbols}

    def OnData(self, data):
        # Update prices
        for symbol in self.symbols:
            if symbol in data and data[symbol].Volume > 0:
                self.prices[symbol].Add(data[symbol].Close)

        if not all(self.prices[s].IsReady for s in self.symbols):
            return

        if not self.initialized:
            self._initialize()
            self.initialized = True
            return

        self.days_since_rebalance += 1
        if self.days_since_rebalance >= self.rebalance_days:
            self._rebalance()
            self.days_since_rebalance = 0

    def _initialize(self):
        """Initialize portfolio with equal weights"""
        self.current_weights = {s: 1.0 / len(self.symbols) for s in self.symbols}
        self._apply_weights()
        self.Debug("Initialized equal-weight portfolio")

    def _rebalance(self):
        """Rebalance portfolio using residual state engine"""
        # Build return matrix
        returns_matrix = self._build_return_matrix()
        
        if returns_matrix is None or len(returns_matrix) < 10:
            self.Debug("Insufficient data for rebalancing")
            return

        # Calculate optimal weights
        new_weights = self._calculate_optimal_weights(returns_matrix)
        
        # Update positions
        self.current_weights = new_weights
        self._apply_weights()
        self.Debug(f"Rebalanced portfolio: {new_weights}")

    def _build_return_matrix(self):
        """Build return matrix from price data"""
        returns = []
        
        for i in range(1, self.prices[self.symbols[0]].Count):
            day_returns = []
            for symbol in self.symbols:
                price_new = self.prices[symbol][i - 1]
                price_old = self.prices[symbol][i]
                
                if price_old > 0:
                    ret = (price_new - price_old) / price_old
                    day_returns.append(ret)
                else:
                    return None

            if len(day_returns) == len(self.symbols):
                returns.append(day_returns)

        return np.array(returns) if returns else None

    def _calculate_optimal_weights(self, returns_matrix):
        """Calculate optimal weights using PCA and energy framework"""
        try:
            # Calculate covariance matrix
            cov_matrix = np.cov(returns_matrix.T)
            
            # Simple weight calculation
            mean_returns = np.mean(returns_matrix, axis=0)
            
            # Regularized covariance
            identity = np.eye(len(self.symbols))
            reg_cov = cov_matrix + self.lambda_l2 * identity
            
            # Inverse for weight calculation
            try:
                inv_cov = np.linalg.inv(reg_cov)
                weights = inv_cov @ mean_returns
            except:
                weights = np.ones(len(self.symbols))

            # Normalize to sum to 1
            weights = np.abs(weights)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(len(self.symbols)) / len(self.symbols)

            # Map to symbols
            result = {}
            for i, symbol in enumerate(self.symbols):
                result[symbol] = float(weights[i])

            return result

        except Exception as e:
            self.Debug(f"Error calculating weights: {str(e)}")
            return {s: 1.0 / len(self.symbols) for s in self.symbols}

    def _apply_weights(self):
        """Apply calculated weights to portfolio"""
        total_value = self.Portfolio.TotalPortfolioValue
        
        for symbol in self.symbols:
            target_value = total_value * self.current_weights[symbol]
            current_value = self.Portfolio[symbol].Holdings.TotalCloseProfit + \
                           (self.Portfolio[symbol].Quantity * self.Securities[symbol].Price)
            
            diff = target_value - current_value
            
            if abs(diff) > 100:  # Only rebalance if difference is significant
                price = self.Securities[symbol].Price
                target_qty = int(target_value / price) if price > 0 else 0
                current_qty = self.Portfolio[symbol].Quantity
                qty_diff = target_qty - current_qty
                
                if qty_diff > 0:
                    self.Buy(symbol, qty_diff)
                elif qty_diff < 0:
                    self.Sell(symbol, -qty_diff)

    def OnEndOfDay(self):
        """Log portfolio metrics"""
        self.Plot("Portfolio", "Value", self.Portfolio.TotalPortfolioValue)
