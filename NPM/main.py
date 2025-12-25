from AlgorithmImports import *
import numpy as np
from collections import deque


class MomentumRotationStrategy(QCAlgorithm):
    """
    Estrategia de Rotacion por Momentum Cross-Sectional
    
    Concepto simple y probado academicamente:
    1. Universe de ETFs diversificados
    2. Ranking por momentum (12-1 month)
    3. Comprar los top N, evitar los bottom M
    4. Filtro de regimen: si SPY < SMA200, ir a bonds/cash
    5. Rebalanceo mensual (no mas frecuente)
    """
    
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(100000)
        
        # UNIVERSE
        self.risk_tickers = [
            "SPY", "QQQ", "IWM", "EFA", "EEM",
            "XLF", "XLE", "XLK", "XLV", "XLI",
        ]
        
        self.safe_tickers = ["TLT", "IEF", "GLD"]
        self.all_tickers = self.risk_tickers + self.safe_tickers
        
        self.symbols = {}
        for ticker in self.all_tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
        
        self.set_benchmark(self.symbols["SPY"])
        
        # PARAMETROS
        self.lookback_long = 252
        self.lookback_short = 21
        self.top_n = 4
        self.equal_weight = True
        self.rebalance_days = 21
        self.regime_sma = 200
        self.regime_lookback = 10
        self.max_position = 0.30
        self.cash_buffer = 0.05
        
        # DATA BUFFERS
        self.prices = {t: deque(maxlen=self.lookback_long + 50) for t in self.all_tickers}
        self.regime_history = deque(maxlen=self.regime_lookback)
        self.last_rebalance = None
        
        self.set_warm_up(self.lookback_long + 30)
        self.debug("=== MOMENTUM ROTATION STRATEGY ===")

    def on_data(self, data: Slice):
        self._update_prices(data)
        
        if self.is_warming_up:
            return
        
        if not self._has_enough_data():
            return
        
        self._update_regime()
        
        t = self.time
        if self.last_rebalance is None or (t - self.last_rebalance).days >= self.rebalance_days:
            self._rebalance()
            self.last_rebalance = t

    def _update_prices(self, data: Slice):
        for ticker in self.all_tickers:
            symbol = self.symbols[ticker]
            if symbol in data and data[symbol] is not None:
                self.prices[ticker].append(float(data[symbol].Close))
    
    def _has_enough_data(self):
        min_len = min(len(self.prices[t]) for t in self.all_tickers)
        return min_len >= self.lookback_long
    
    def _update_regime(self):
        spy_prices = np.array(self.prices["SPY"])
        if len(spy_prices) < self.regime_sma:
            self.regime_history.append(1)
            return
        
        sma200 = spy_prices[-self.regime_sma:].mean()
        current_price = spy_prices[-1]
        regime = 1 if current_price > sma200 else 0
        self.regime_history.append(regime)

    def _calculate_momentum(self, ticker):
        prices = np.array(self.prices[ticker])
        if len(prices) < self.lookback_long:
            return -999.0
        
        price_12m = prices[-self.lookback_long]
        price_1m = prices[-self.lookback_short]
        momentum = (price_1m - price_12m) / price_12m
        return momentum
    
    def _calculate_volatility(self, ticker, window=60):
        prices = np.array(self.prices[ticker])
        if len(prices) < window + 1:
            return 0.20
        
        returns = np.diff(np.log(prices[-window-1:]))
        vol = returns.std() * np.sqrt(252)
        return max(vol, 0.05)

    def _rebalance(self):
        regime = self._get_current_regime()
        
        if regime == "risk-off":
            self._allocate_defensive()
        else:
            self._allocate_momentum()
    
    def _get_current_regime(self):
        if len(self.regime_history) < self.regime_lookback:
            return "risk-on"
        
        risk_off_days = sum(1 for r in self.regime_history if r == 0)
        if risk_off_days >= self.regime_lookback * 0.7:
            return "risk-off"
        
        return "risk-on"
    
    def _allocate_defensive(self):
        self.debug(f"RISK-OFF @ {self.time.date()}")
        
        for ticker in self.risk_tickers:
            self.set_holdings(self.symbols[ticker], 0)
        
        safe_momentum = {}
        for ticker in self.safe_tickers:
            safe_momentum[ticker] = self._calculate_momentum(ticker)
        
        ranked = sorted(self.safe_tickers, key=lambda t: safe_momentum[t], reverse=True)
        allocable = 1.0 - self.cash_buffer
        weights = [0.50, 0.30, 0.15]
        
        for i, ticker in enumerate(ranked):
            if i < len(weights):
                w = weights[i] * allocable
                self.set_holdings(self.symbols[ticker], w)
    
    def _allocate_momentum(self):
        self.debug(f"RISK-ON @ {self.time.date()}")
        
        for ticker in self.safe_tickers:
            if ticker != "GLD":
                self.set_holdings(self.symbols[ticker], 0)
        
        risk_momentum = {}
        for ticker in self.risk_tickers:
            mom = self._calculate_momentum(ticker)
            vol = self._calculate_volatility(ticker)
            risk_momentum[ticker] = mom / vol if vol > 0 else mom
        
        ranked = sorted(self.risk_tickers, key=lambda t: risk_momentum[t], reverse=True)
        selected = ranked[:self.top_n]
        selected = [t for t in selected if risk_momentum[t] > 0]
        
        if not selected:
            self._go_to_cash()
            return
        
        allocable = 1.0 - self.cash_buffer - 0.05
        weight_per_asset = allocable / len(selected)
        
        for ticker in self.risk_tickers:
            if ticker in selected:
                w = min(weight_per_asset, self.max_position)
                self.set_holdings(self.symbols[ticker], w)
            else:
                self.set_holdings(self.symbols[ticker], 0)
        
        self.set_holdings(self.symbols["GLD"], 0.05)
    
    def _go_to_cash(self):
        for ticker in self.risk_tickers:
            self.set_holdings(self.symbols[ticker], 0)
        
        self.set_holdings(self.symbols["TLT"], 0.30)
        self.set_holdings(self.symbols["IEF"], 0.20)
        self.set_holdings(self.symbols["GLD"], 0.10)

