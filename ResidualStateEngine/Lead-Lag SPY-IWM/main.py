# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
# endregion


class LeadLagSPYIWM(QCAlgorithm):
    """
    Lead-Lag Strategy: SPY leads IWM
    
    Explota el lag estructural entre large caps (SPY) y small caps (IWM).
    SPY reacciona más rápido a información de mercado debido a mayor liquidez
    y cobertura institucional.
    
    Lógica:
    1. Monitorear retornos de SPY en ventana rolling
    2. Calcular z-score del retorno
    3. Si z-score > threshold → LONG IWM (SPY subió, IWM seguirá)
    4. Si z-score < -threshold → SHORT IWM (SPY bajó, IWM seguirá)
    5. Cerrar posición después del lag estimado
    """

    def initialize(self):
        # ============== PARÁMETROS DE BACKTEST ==============
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 1)
        self.set_cash(100000)
        
        # ============== ASSETS ==============
        self.leader = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.lagger = self.add_equity("IWM", Resolution.MINUTE).symbol
        
        # ============== PARÁMETROS DE ESTRATEGIA ==============
        # Ventana para calcular retorno del leader (minutos)
        self.lookback_window = 15
        
        # Lag esperado entre SPY e IWM (minutos)
        # Esto se puede calibrar dinámicamente
        self.expected_lag = 10
        
        # Threshold en z-score para entrar
        self.entry_threshold = 2.0
        
        # Ventana para calcular media/std del retorno (para z-score)
        self.zscore_lookback = 60  # minutos
        
        # Máximo de posiciones simultáneas
        self.max_positions = 3
        
        # Sizing por trade (fracción del portfolio)
        self.position_size = 0.25
        
        # ============== ESTADO ==============
        # Buffer de precios del leader
        self.leader_prices = deque(maxlen=self.lookback_window + 1)
        
        # Buffer de retornos históricos para z-score
        self.historical_returns = deque(maxlen=self.zscore_lookback)
        
        # Trades abiertos: lista de (entry_time, direction, shares)
        self.open_trades = []
        
        # Filtro de horario: solo operar en horas líquidas
        self.market_open_buffer = 30  # minutos después de apertura
        self.market_close_buffer = 30  # minutos antes de cierre
        
        # ============== CALIBRACIÓN DINÁMICA ==============
        self.calibration_mode = False
        self.calibration_period = 20  # días para calibrar
        self.days_since_start = 0
        
        # Cross-correlations para encontrar lag óptimo
        self.leader_returns_buffer = deque(maxlen=500)
        self.lagger_returns_buffer = deque(maxlen=500)
        
        # ============== MÉTRICAS ==============
        self.total_trades = 0
        self.winning_trades = 0
        
        # Schedule recalibración semanal
        self.schedule.on(
            self.date_rules.every(DayOfWeek.MONDAY),
            self.time_rules.after_market_open(self.leader, 60),
            self.recalibrate_lag
        )
        
        # Warm-up para tener suficiente data
        self.set_warm_up(timedelta(days=5))

    def on_data(self, data: Slice):
        """Procesa cada barra de minuto"""
        
        if self.is_warming_up:
            return
            
        # Verificar que tenemos data de ambos activos
        if not (data.contains_key(self.leader) and data.contains_key(self.lagger)):
            return
            
        if not (data[self.leader] and data[self.lagger]):
            return
        
        current_time = self.time
        leader_price = data[self.leader].close
        lagger_price = data[self.lagger].close
        
        # Actualizar buffers de precios
        self.leader_prices.append(leader_price)
        
        # Guardar retornos para calibración
        if len(self.leader_prices) >= 2:
            leader_ret = (leader_price / self.leader_prices[-2]) - 1
            lagger_ret = (lagger_price / data[self.lagger].open) - 1 if data[self.lagger].open > 0 else 0
            
            self.leader_returns_buffer.append(leader_ret)
            self.lagger_returns_buffer.append(lagger_ret)
        
        # Filtro de horario de mercado
        if not self._is_trading_hours(current_time):
            return
        
        # ============== CERRAR TRADES EXPIRADOS ==============
        self._close_expired_trades(current_time, lagger_price)
        
        # ============== GENERAR SEÑALES ==============
        signal = self._calculate_signal()
        
        if signal != 0 and len(self.open_trades) < self.max_positions:
            self._enter_trade(signal, current_time, lagger_price)

    def _is_trading_hours(self, current_time):
        """Filtrar horarios de baja liquidez"""
        market_open = current_time.replace(hour=9, minute=30, second=0)
        market_close = current_time.replace(hour=16, minute=0, second=0)
        
        # Buffer después de apertura
        if current_time < market_open + timedelta(minutes=self.market_open_buffer):
            return False
        
        # Buffer antes de cierre
        if current_time > market_close - timedelta(minutes=self.market_close_buffer):
            return False
            
        return True

    def _calculate_signal(self) -> int:
        """
        Calcula señal de trading basada en z-score del retorno del leader
        
        Returns:
            1 para LONG, -1 para SHORT, 0 para no hacer nada
        """
        if len(self.leader_prices) < self.lookback_window + 1:
            return 0
        
        # Retorno del leader en la ventana
        current_return = (self.leader_prices[-1] / self.leader_prices[0]) - 1
        
        # Actualizar histórico de retornos
        self.historical_returns.append(current_return)
        
        if len(self.historical_returns) < self.zscore_lookback // 2:
            return 0
        
        # Calcular z-score
        returns_array = np.array(self.historical_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return < 1e-8:  # Evitar división por cero
            return 0
        
        z_score = (current_return - mean_return) / std_return
        
        # Generar señal
        if z_score > self.entry_threshold:
            return 1  # LONG: SPY subió fuerte, IWM seguirá
        elif z_score < -self.entry_threshold:
            return -1  # SHORT: SPY bajó fuerte, IWM seguirá
        
        return 0

    def _enter_trade(self, direction: int, entry_time, current_price):
        """Entra en un trade"""
        
        # Calcular shares basado en position size
        portfolio_value = self.portfolio.total_portfolio_value
        trade_value = portfolio_value * self.position_size
        shares = int(trade_value / current_price) * direction
        
        if shares == 0:
            return
        
        # Ejecutar orden
        self.market_order(self.lagger, shares)
        
        # Registrar trade
        self.open_trades.append({
            'entry_time': entry_time,
            'direction': direction,
            'shares': shares,
            'entry_price': current_price,
            'exit_time': entry_time + timedelta(minutes=self.expected_lag)
        })
        
        self.total_trades += 1
        
        self.debug(f"ENTRY: {'LONG' if direction > 0 else 'SHORT'} {abs(shares)} IWM @ {current_price:.2f}")

    def _close_expired_trades(self, current_time, current_price):
        """Cierra trades que han alcanzado su tiempo de holding"""
        
        trades_to_close = []
        
        for i, trade in enumerate(self.open_trades):
            if current_time >= trade['exit_time']:
                trades_to_close.append(i)
                
                # Cerrar posición
                self.market_order(self.lagger, -trade['shares'])
                
                # Calcular P&L
                pnl = (current_price - trade['entry_price']) * trade['shares']
                if pnl > 0:
                    self.winning_trades += 1
                
                self.debug(f"EXIT: {abs(trade['shares'])} IWM @ {current_price:.2f}, PnL: ${pnl:.2f}")
        
        # Remover trades cerrados (en orden inverso para no joder índices)
        for i in reversed(trades_to_close):
            self.open_trades.pop(i)

    def recalibrate_lag(self):
        """
        Recalibra el lag óptimo usando cross-correlation
        Se ejecuta semanalmente
        """
        if len(self.leader_returns_buffer) < 200:
            return
        
        leader_rets = np.array(self.leader_returns_buffer)
        lagger_rets = np.array(self.lagger_returns_buffer)
        
        # Buscar lag con máxima correlación
        best_lag = self.expected_lag
        best_corr = 0
        
        for lag in range(1, 31):  # Testear lags de 1 a 30 minutos
            if lag >= len(leader_rets):
                break
                
            # Correlación: leader_t vs lagger_{t+lag}
            leader_shifted = leader_rets[:-lag]
            lagger_future = lagger_rets[lag:]
            
            if len(leader_shifted) < 50:
                continue
            
            corr = np.corrcoef(leader_shifted, lagger_future)[0, 1]
            
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag
        
        old_lag = self.expected_lag
        self.expected_lag = best_lag
        
        self.debug(f"RECALIBRATION: Lag {old_lag} -> {best_lag} (corr: {best_corr:.4f})")

    def on_end_of_day(self, symbol):
        """Cierra todas las posiciones al final del día"""
        if symbol == self.lagger:
            # Cerrar todos los trades abiertos
            for trade in self.open_trades:
                self.market_order(self.lagger, -trade['shares'])
            self.open_trades = []
            
            # Asegurar que no quedamos con posición
            if self.portfolio[self.lagger].invested:
                self.liquidate(self.lagger)

    def on_end_of_algorithm(self):
        """Estadísticas finales"""
        win_rate = self.winning_trades / max(1, self.total_trades) * 100
        
        self.log(f"========== RESULTADOS FINALES ==========")
        self.log(f"Total Trades: {self.total_trades}")
        self.log(f"Winning Trades: {self.winning_trades}")
        self.log(f"Win Rate: {win_rate:.1f}%")
        self.log(f"Final Lag Setting: {self.expected_lag} minutes")

