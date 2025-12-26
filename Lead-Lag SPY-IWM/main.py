# region imports
from AlgorithmImports import *
# endregion

class LeadLagSPYIWM(QCAlgorithm):
    """SPY-IWM lead-lag trading strategy"""

    def initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        self.AddEquity("SPY", Resolution.Minute)
        self.AddEquity("IWM", Resolution.Minute)
        
        self.lead = "SPY"
        self.lag = "IWM"
        
        # Position tracking
        self.positions = {}
        self.max_positions = 3
        self.position_holding_period = {}
        
        # Lag calculation
        self.last_lag_calc = None
        self.optimal_lag = 10  # default 10 minutes
        self.lookback = 60  # use 1 hour of data
        
    def OnData(self, data):
        if not data.HasData:
            return

        # Market hours check (9:30 AM - 4:00 PM ET with buffers)
        if not self._is_market_hours(self.Time):
            return

        # Recalculate lag weekly
        if self.last_lag_calc is None or (self.Time - self.last_lag_calc).days >= 5:
            self._calculate_lag()
            self.last_lag_calc = self.Time

        # Check for entry signals
        if self.lead in data and self.lag in data:
            lead_price = data[self.lead].Close
            lag_price = data[self.lag].Close
            
            # Simple momentum signal on lead
            if self._should_enter(lead_price):
                self._enter_position(lag_price)

        # Manage positions
        self._manage_positions()

    def _is_market_hours(self, time):
        """Check if we're during market hours with buffers"""
        hour = time.hour
        minute = time.minute
        
        # Start: 9:35 AM (buffer after 9:30 open)
        # End: 3:55 PM (buffer before 4:00 close)
        is_during_hours = (9 < hour < 16) or (hour == 9 and minute >= 35) or (hour == 15 and minute < 55)
        return is_during_hours

    def _calculate_lag(self):
        """Calculate optimal lag between SPY and IWM"""
        history = self.History([self.lead, self.lag], self.lookback, Resolution.Minute)
        
        if len(history) < 10:
            self.optimal_lag = 10
            return

        lead_prices = history[self.lead]["close"].values
        lag_prices = history[self.lag]["close"].values
        
        if len(lead_prices) < 2 or len(lag_prices) < 2:
            return

        # Simple lag detection
        lead_returns = [(lead_prices[i] - lead_prices[i-1]) / lead_prices[i-1] 
                        for i in range(1, len(lead_prices))]
        lag_returns = [(lag_prices[i] - lag_prices[i-1]) / lag_prices[i-1] 
                       for i in range(1, len(lag_prices))]

        max_corr = 0
        best_lag = 10

        for lag in range(0, 20):
            if lag >= len(lead_returns) or lag >= len(lag_returns):
                break
                
            lead_subset = lead_returns[:-lag] if lag > 0 else lead_returns
            lag_subset = lag_returns[lag:] if lag > 0 else lag_returns
            
            if len(lead_subset) > 2 and len(lag_subset) > 2:
                corr = self._calc_corr(lead_subset, lag_subset)
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag

        self.optimal_lag = best_lag
        self.Debug(f"Updated optimal lag to {best_lag} minutes")

    def _calc_corr(self, x, y):
        """Calculate correlation"""
        n = min(len(x), len(y))
        if n < 2:
            return 0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den_x = (sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5)
        den_y = (sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5)
        
        if den_x * den_y == 0:
            return 0
        return num / (den_x * den_y)

    def _should_enter(self, lead_price):
        """Check if we should enter a new position"""
        if len(self.positions) >= self.max_positions:
            return False

        # Get recent history
        history = self.History([self.lead], 5, Resolution.Minute)
        if len(history) < 3:
            return False

        prices = history[self.lead]["close"].values
        recent_return = (prices[-1] - prices[0]) / prices[0]

        return recent_return > 0.001  # 0.1% threshold

    def _enter_position(self, entry_price):
        """Enter a new position"""
        position_id = len(self.positions)
        qty = int(self.Portfolio.Cash / (self.max_positions * entry_price))
        
        if qty > 0:
            self.Buy(self.lag, qty)
            self.positions[position_id] = {
                'entry_price': entry_price,
                'entry_time': self.Time,
                'qty': qty
            }
            self.position_holding_period[position_id] = self.Time

    def _manage_positions(self):
        """Close positions after holding period"""
        to_close = []
        
        for pos_id, pos in self.positions.items():
            holding_time = (self.Time - pos['entry_time']).total_seconds() / 60
            
            if holding_time > self.optimal_lag:
                to_close.append(pos_id)

        for pos_id in to_close:
            self.Liquidate(self.lag)
            del self.positions[pos_id]
            if pos_id in self.position_holding_period:
                del self.position_holding_period[pos_id]
