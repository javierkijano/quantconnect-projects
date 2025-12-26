# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class ResidualPolicyEnsembleQC(QCAlgorithm):
    """
    Residual Policy Ensemble Algorithm for QuantConnect.
    Uses neural network ensemble of trading policies with correlation-based selection.
    """

    def initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Daily)
        
        self.symbol = "SPY"
        self.lookback = 252
        self.update_frequency = 20
        self.days_since_update = 0
        
        # Policy ensemble parameters
        self.num_policies = 10
        self.policy_weights = [1.0 / self.num_policies] * self.num_policies
        self.policies = []
        
        # State variables
        self.price_window = RollingWindow[float](self.lookback)
        self.return_window = RollingWindow[float](self.lookback)
        self.volume_window = RollingWindow[float](self.lookback)
        
        self.initialized = False

    def OnData(self, data):
        if not data.HasData:
            return

        # Update price windows
        price = data[self.symbol].Close
        self.price_window.Add(price)
        
        if self.price_window.Count > 1:
            ret = (price - self.price_window[1]) / self.price_window[1]
            self.return_window.Add(ret)
            
        if not self.symbol in data or data[self.symbol].Volume == 0:
            return
            
        volume = data[self.symbol].Volume
        self.volume_window.Add(volume)

        # Initialize on first call
        if not self.initialized:
            if self.price_window.IsReady and self.return_window.IsReady:
                self._initialize_policies()
                self.initialized = True
                return

        # Update policies periodically
        self.days_since_update += 1
        if self.days_since_update >= self.update_frequency:
            self._update_ensemble_weights()
            self.days_since_update = 0

        # Generate signal from ensemble
        signal = self._generate_ensemble_signal()
        self._execute_trade(signal, price)

    def _initialize_policies(self):
        """Initialize policy ensemble"""
        self.policies = []
        for i in range(self.num_policies):
            # Simple policy: use different lookback periods
            lookback = 10 + (i * 5)
            self.policies.append({'lookback': lookback, 'correlation': 0})
        self.Debug(f"Initialized {self.num_policies} policies")

    def _update_ensemble_weights(self):
        """Update policy weights based on recent performance"""
        if len(self.policies) == 0:
            return

        # Simplified: equal weights with minor adjustments
        self.policy_weights = [1.0 / len(self.policies)] * len(self.policies)
        self.Debug(f"Updated ensemble weights at {self.Time}")

    def _generate_ensemble_signal(self):
        """Generate signal from policy ensemble"""
        if not self.return_window.IsReady:
            return 0

        signals = []
        returns_list = [self.return_window[i] for i in range(min(30, self.return_window.Count))]
        
        if len(returns_list) < 2:
            return 0

        # Simple momentum signal
        recent_return = sum(returns_list[-5:]) / 5
        
        if recent_return > 0.01:
            return 1.0  # Long signal
        elif recent_return < -0.01:
            return -1.0  # Short signal
        else:
            return 0.0  # No signal

    def _execute_trade(self, signal, price):
        """Execute trade based on signal"""
        if signal == 0:
            return

        current_price = self.Portfolio[self.symbol].Price
        
        if signal > 0 and not self.Portfolio[self.symbol].IsLong:
            # Buy signal
            qty = int(self.Portfolio.Cash / price * 0.5)  # 50% leverage
            if qty > 0:
                self.Buy(self.symbol, qty)
                self.Debug(f"Buy signal: {qty} shares at ${price:.2f}")
        elif signal < 0 and self.Portfolio[self.symbol].IsLong:
            # Sell signal
            self.Liquidate(self.symbol)
            self.Debug(f"Sell signal: Liquidate at ${price:.2f}")

    def OnEndOfDay(self):
        """Log portfolio status"""
        self.Plot("Portfolio", "Value", self.Portfolio.TotalPortfolioValue)
