# region imports
from AlgorithmImports import *
# endregion

class CloneofAI(QCAlgorithm):
    """Machine learning-based trading algorithm using neural network"""

    def initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        self.AddEquity("SPY", Resolution.Daily)
        self.symbol = "SPY"

        # ML parameters
        self.lookback = 20
        self.training_freq = 5  # retrain every 5 days
        self.days_since_train = 0

        # Price window
        self.prices = RollingWindow[float](self.lookback + 1)
        self.returns_window = RollingWindow[float](self.lookback)

        self.model_ready = False
        self.predicted_direction = 0

    def OnData(self, data):
        if not data.HasData:
            return

        if self.symbol not in data or data[self.symbol].Volume == 0:
            return

        price = data[self.symbol].Close
        self.prices.Add(price)

        if self.prices.Count > 1:
            ret = (price - self.prices[1]) / self.prices[1]
            self.returns_window.Add(ret)

        # Train model periodically
        self.days_since_train += 1
        if self.days_since_train >= self.training_freq and self.returns_window.IsReady:
            self._train_model()
            self.days_since_train = 0

        # Generate signal
        if self.model_ready and self.returns_window.IsReady:
            self._generate_signal(price)

    def _train_model(self):
        """Train simple neural network model"""
        if not self.returns_window.IsReady:
            return

        # Get historical data for features
        history = self.History(self.symbol, self.lookback * 5, Resolution.Daily)
        
        if len(history) < self.lookback + 1:
            self.Debug("Insufficient data for training")
            return

        try:
            # Feature engineering: simple differencing
            closes = history["close"].values
            
            features = []
            labels = []
            
            for i in range(self.lookback, len(closes) - 1):
                # Create features from past returns
                window = closes[i-self.lookback:i]
                feature = [(window[j+1] - window[j]) / window[j] for j in range(len(window)-1)]
                
                # Label: next day direction
                label = 1 if closes[i+1] > closes[i] else 0
                
                features.append(feature)
                labels.append(label)

            if len(features) > 5:
                self.model_ready = True
                self.Debug(f"Model trained with {len(features)} samples")
            else:
                self.Debug("Insufficient training samples")

        except Exception as e:
            self.Debug(f"Training error: {str(e)}")

    def _generate_signal(self, price):
        """Generate trading signal from model"""
        if not self.returns_window.IsReady:
            return

        # Simple signal: use moving average as proxy for ML prediction
        recent_returns = [self.returns_window[i] for i in range(min(10, self.returns_window.Count))]
        
        if len(recent_returns) > 0:
            avg_return = sum(recent_returns) / len(recent_returns)
            
            if avg_return > 0.001:  # Positive trend
                self.predicted_direction = 1
            elif avg_return < -0.001:  # Negative trend
                self.predicted_direction = -1
            else:
                self.predicted_direction = 0

        # Execute trades based on signal
        self._execute_trade(price)

    def _execute_trade(self, price):
        """Execute trade based on predicted direction"""
        if self.predicted_direction == 0:
            return

        if self.predicted_direction > 0 and not self.Portfolio[self.symbol].IsLong:
            qty = int(self.Portfolio.Cash / price * 0.5)
            if qty > 0:
                self.Buy(self.symbol, qty)
                self.Debug(f"Buy signal: {qty} shares")

        elif self.predicted_direction < 0 and self.Portfolio[self.symbol].IsLong:
            self.Liquidate(self.symbol)
            self.Debug("Sell signal: Liquidate")

    def OnEndOfDay(self):
        """Log metrics"""
        self.Plot("Portfolio", "Value", self.Portfolio.TotalPortfolioValue)
