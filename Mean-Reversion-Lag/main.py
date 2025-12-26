# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
from scipy import signal
# endregion


class MeanReversionLag(QCAlgorithm):
    """
    Lead-Lag Strategy con Cross-Correlation Dinámica
    
    Detecta el lag óptimo entre SPY e IWM cada hora usando cross-correlation
    y ajusta la estrategia según el tipo de correlación encontrada
    """

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 1)
        self.set_cash(100000)
        
        self.leader = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.lagger = self.add_equity("IWM", Resolution.MINUTE).symbol
        
        # Cross-correlation params
        self.max_lag_test = 30
        self.min_correlation = 0.6
        self.recalc_interval = 60
        self.last_recalc_time = None
        
        # Dynamic lag results
        self.optimal_lag = None
        self.optimal_correlation = 0
        self.correlation_type = None
        
        # Signal params
        self.lookback_window = 15
        self.entry_threshold = 2.0
        self.zscore_lookback = 120
        
        # Position params
        self.max_positions = 1
        self.position_size = 0.20
        self.trade_cooldown = 60
        self.last_trade_time = None
        
        # Data buffers
        self.leader_prices = deque(maxlen=self.lookback_window + 1)
        self.historical_returns = deque(maxlen=self.zscore_lookback)
        self.open_trades = []
        
        # Cross-correlation buffers
        self.cross_corr_window = 200
        self.leader_returns_buffer = deque(maxlen=self.cross_corr_window)
        self.lagger_returns_buffer = deque(maxlen=self.cross_corr_window)
        self.last_leader_price = None
        self.last_lagger_price = None
        
        # Market hours
        self.market_open_buffer = 45
        self.market_close_buffer = 45
        
        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        
        self.set_warm_up(timedelta(days=5))

    def on_data(self, data: Slice):
        if self.is_warming_up:
            return
            
        if not (data.contains_key(self.leader) and data.contains_key(self.lagger)):
            return
            
        if not (data[self.leader] and data[self.lagger]):
            return
        
        current_time = self.time
        leader_price = data[self.leader].close
        lagger_price = data[self.lagger].close
        
        # Update return buffers
        if self.last_leader_price is not None and self.last_lagger_price is not None:
            leader_ret = (leader_price - self.last_leader_price) / self.last_leader_price
            lagger_ret = (lagger_price - self.last_lagger_price) / self.last_lagger_price
            self.leader_returns_buffer.append(leader_ret)
            self.lagger_returns_buffer.append(lagger_ret)
        
        self.last_leader_price = leader_price
        self.last_lagger_price = lagger_price
        self.leader_prices.append(leader_price)
        
        self._manage_open_trades(current_time, lagger_price)
        
        if not self._is_trading_time(current_time):
            return
        
        if self.last_trade_time is not None:
            minutes_since_last = (current_time - self.last_trade_time).total_seconds() / 60
            if minutes_since_last < self.trade_cooldown:
                return
        
        # Recalculate optimal lag
        self._update_optimal_lag(current_time)
        
        # Check if we have valid lag
        if self.optimal_lag is None:
            return
        
        if abs(self.optimal_correlation) < self.min_correlation:
            return
        
        if len(self.leader_prices) < self.lookback_window + 1:
            return
            
        leader_return = (self.leader_prices[-1] - self.leader_prices[0]) / self.leader_prices[0]
        self.historical_returns.append(leader_return)
        
        if len(self.historical_returns) < 30:
            return
        
        returns_array = np.array(self.historical_returns)
        mean_ret = np.mean(returns_array)
        std_ret = np.std(returns_array)
        
        if std_ret < 1e-8:
            return
            
        zscore = (leader_return - mean_ret) / std_ret
        
        if len(self.open_trades) >= self.max_positions:
            return
        
        # Dynamic signal based on correlation type
        if self.correlation_type == 'negative':
            # Mean reversion
            if zscore > self.entry_threshold:
                self._open_trade(current_time, lagger_price, -1)
            elif zscore < -self.entry_threshold:
                self._open_trade(current_time, lagger_price, 1)
        elif self.correlation_type == 'positive':
            # Momentum
            if zscore > self.entry_threshold:
                self._open_trade(current_time, lagger_price, 1)
            elif zscore < -self.entry_threshold:
                self._open_trade(current_time, lagger_price, -1)
    
    def _open_trade(self, time, price, direction):
        shares = int(self.portfolio.cash * self.position_size / price) * direction
        
        if shares == 0:
            return
            
        self.market_order(self.lagger, shares)
        
        self.open_trades.append({
            'entry_time': time,
            'entry_price': price,
            'shares': shares,
            'direction': direction
        })
        
        self.last_trade_time = time
        self.total_trades += 1
        
        direction_str = "LONG" if direction > 0 else "SHORT"
        self.debug(f"{time} - {direction_str} IWM @ {price:.2f} | Lag: {self.optimal_lag}min | Corr: {self.optimal_correlation:.2f}")
    
    def _manage_open_trades(self, current_time, current_price):
        trades_to_close = []
        exit_lag = self.optimal_lag if self.optimal_lag else 15
        
        for i, trade in enumerate(self.open_trades):
            minutes_held = (current_time - trade['entry_time']).total_seconds() / 60
            
            if minutes_held >= exit_lag:
                self.market_order(self.lagger, -trade['shares'])
                
                if trade['direction'] > 0:
                    pnl = current_price - trade['entry_price']
                else:
                    pnl = trade['entry_price'] - current_price
                
                if pnl > 0:
                    self.winning_trades += 1
                
                trades_to_close.append(i)
        
        for i in reversed(trades_to_close):
            self.open_trades.pop(i)
    
    def _update_optimal_lag(self, current_time):
        """Calculate optimal lag using cross-correlation"""
        if self.last_recalc_time is not None:
            minutes_since = (current_time - self.last_recalc_time).total_seconds() / 60
            if minutes_since < self.recalc_interval:
                return
        
        if len(self.leader_returns_buffer) < 100:
            return
        
        try:
            leader_rets = np.array(self.leader_returns_buffer)
            lagger_rets = np.array(self.lagger_returns_buffer)
            
            max_lag = min(self.max_lag_test, len(leader_rets) // 3)
            correlations = signal.correlate(lagger_rets, leader_rets, mode='same')
            lags = signal.correlation_lags(len(lagger_rets), len(leader_rets), mode='same')
            
            # Only positive lags (SPY leads IWM)
            positive_lag_mask = (lags > 0) & (lags <= max_lag)
            
            if not np.any(positive_lag_mask):
                return
            
            valid_correlations = correlations[positive_lag_mask]
            valid_lags = lags[positive_lag_mask]
            
            # Normalize
            valid_correlations = valid_correlations / (len(leader_rets) * np.std(leader_rets) * np.std(lagger_rets))
            
            # Find best lag
            max_corr_idx = np.argmax(np.abs(valid_correlations))
            best_lag = int(valid_lags[max_corr_idx])
            best_correlation = valid_correlations[max_corr_idx]
            
            if abs(best_correlation) > 0.3:
                self.optimal_lag = best_lag
                self.optimal_correlation = best_correlation
                self.correlation_type = 'positive' if best_correlation > 0 else 'negative'
                
                self.log(f"{current_time} - Lag óptimo: {best_lag}min | Correlación: {best_correlation:.3f} ({self.correlation_type})")
            
            self.last_recalc_time = current_time
            
        except Exception as e:
            self.debug(f"Error calculando lag: {e}")
    
    def _is_trading_time(self, time):
        market_open = time.replace(hour=9, minute=30, second=0)
        market_close = time.replace(hour=16, minute=0, second=0)
        
        valid_start = market_open + timedelta(minutes=self.market_open_buffer)
        valid_end = market_close - timedelta(minutes=self.market_close_buffer)
        
        return valid_start <= time <= valid_end

    def on_end_of_day(self, symbol):
        if symbol == self.lagger:
            for trade in self.open_trades:
                self.market_order(self.lagger, -trade['shares'])
            self.open_trades = []
            
            if self.portfolio[self.lagger].invested:
                self.liquidate(self.lagger)

    def on_end_of_algorithm(self):
        win_rate = self.winning_trades / max(1, self.total_trades) * 100
        
        self.log(f"========== RESULTADOS FINALES ==========")
        self.log(f"Total Trades: {self.total_trades}")
        self.log(f"Winning Trades: {self.winning_trades}")
        self.log(f"Win Rate: {win_rate:.1f}%")
        self.log(f"Lag óptimo final: {self.optimal_lag}min")
        self.log(f"Correlación final: {self.optimal_correlation:.3f}")

