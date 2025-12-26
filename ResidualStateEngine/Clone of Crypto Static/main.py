from AlgorithmImports import *
import numpy as np
from scipy import stats

class TrianguloDeControl(QCAlgorithm):
    
    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_cash(100000)
        
        # El Triángulo Expandido: ETH (Señal), Multiple Altcoins (Objetivos), BTC (Control)
        self._eth = self.add_crypto("ETHUSD", Resolution.HOUR).symbol

        
        # Cascada de Liquidez: ETH (Leader) → SOL (Beta Apalancado) → Memes (Beta Extremo)
        self._eth = self.add_crypto("ETHUSD", Resolution.MINUTE, Market.GDAX).symbol
        self._sol = self.add_crypto("SOLUSD", Resolution.MINUTE, Market.GDAX).symbol
        self._btc = self.add_crypto("BTCUSD", Resolution.MINUTE, Market.GDAX).symbol
        
        # Meme coins del ecosistema SOL (beta 3-5x)
        # Nota: Si no tienen datos en GDAX, quitarlos
        self._meme_coins = []  # WIF/BONK no siempre disponibles en todas las exchanges
        
        # Filtro Macro: SMA 200 diario de BTC para detectar bear markets
        self._btc_daily = self.add_crypto("BTCUSD", Resolution.DAILY, Market.GDAX).symbol
        self._sma_200 = self.sma(self._btc_daily, 200, Resolution.DAILY)
        
        # Configuración de ventanas (MINUTOS para capturar lead-lag)
        self._ventana_regresion = 120  # 120 minutos para calcular residuos
        self._ventana_ccf = 60         # 60 minutos para calcular cross-correlation
        self._max_lag = 30             # Buscar lag óptimo hasta 30 minutos
        self._umbral_parcial = 0.5     # Correlación parcial mínima
        self._umbral_eth_puro = 0.001  # 0.1% movimiento puro de ETH
        
        # Lag óptimo ETH→SOL (se recalcula dinámicamente)
        self._lag_optimo = 5  # Inicializar en 5 minutos
        self._ccf_max = 0     # Correlación máxima en el lag óptimo
        
        # Beta dinámico ETH→SOL
        self._beta_eth_sol = 1.5
        
        # Históricos de precios (minutos)
        self._eth_history = RollingWindow[float](self._ventana_regresion)
        self._sol_history = RollingWindow[float](self._ventana_regresion)
        self._btc_history = RollingWindow[float](self._ventana_regresion)
        
        # Ventana para CCF (debe ser más grande que max_lag)
        self._eth_ccf = RollingWindow[float](self._ventana_ccf + self._max_lag)
        self._sol_ccf = RollingWindow[float](self._ventana_ccf + self._max_lag)
        

        
        # Warm up para llenar ventanas
        self.set_warm_up(self._ventana_regresion, Resolution.HOUR)
        
        # Indicadores para tracking
        self._partial_corr = 0
        self._residuo_eth = 0
        self._residuo_sol = 0
        
        # Gestión de Riesgo
        self._peak_equity = 100000
        self._max_drawdown_permitido = 0.25  # 25%
        self._volatilidad_reciente = 0
        
        # Rebalanceo cada 5 minutos para capturar lead-lag
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(minutes=5)),
            self._rebalance
        )
        
        # Chart para visualizar
        self._setup_charts()
    
    def on_data(self, data):
        if self.is_warming_up:
            return
            
        # Actualizar históricos
        if data.contains_key(self._eth) and data[self._eth] is not None:
            self._eth_history.add(float(data[self._eth].close))
            self._eth_ccf.add(float(data[self._eth].close))
        
        if data.contains_key(self._sol) and data[self._sol] is not None:
            self._sol_history.add(float(data[self._sol].close))
            self._sol_ccf.add(float(data[self._sol].close))
        
        if data.contains_key(self._btc) and data[self._btc] is not None:
            self._btc_history.add(float(data[self._btc].close))
    
    def _rebalance(self):
        # Verificar que tenemos datos suficientes
        if not self._eth_history.is_ready or not self._sol_history.is_ready or not self._btc_history.is_ready:
            return
        
        if not self._eth_ccf.is_ready or not self._sol_ccf.is_ready:
            return
        
        if not self._sma_200.is_ready:
            return
        
        # Filtro 1: Gestión de Drawdown - Stop Loss Global
        equity_actual = self.portfolio.total_portfolio_value
        if equity_actual > self._peak_equity:
            self._peak_equity = equity_actual
        
        drawdown_actual = (self._peak_equity - equity_actual) / self._peak_equity
        
        if drawdown_actual > self._max_drawdown_permitido:
            if self.portfolio.invested:
                self.liquidate()
                self.debug(f"STOP LOSS GLOBAL - Drawdown: {drawdown_actual:.2%}")
            return
        
        # Filtro 2: Tendencia Macro - Solo operar si BTC > SMA 200
        btc_precio = self.securities[self._btc_daily].price
        en_bull_market = btc_precio > self._sma_200.current.value
        
        self.plot("Macro", "BTC Price", btc_precio)
        self.plot("Macro", "SMA 200", self._sma_200.current.value)
        self.plot("Risk", "Drawdown %", drawdown_actual * 100)
        
        if not en_bull_market:
            if self.portfolio.invested:
                self.liquidate()
                self.debug(f"BEAR MARKET - BTC: {btc_precio:.0f} < SMA200: {self._sma_200.current.value:.0f}")
            return
        
        # 1. Calcular correlación parcial ETH-SOL|BTC (estructura base)
        partial_corr, residuo_eth, residuo_sol = self._calcular_senal_limpia()
        
        # 2. Calcular Cross-Correlation Function para encontrar lag óptimo
        self._lag_optimo, self._ccf_max, ccf_values = self._calcular_lag_optimo()
        
        # 3. Detectar si ETH tuvo movimiento fuerte hace 'lag_optimo' minutos
        eth_lagged_signal = self._detectar_señal_lagged()
        
        # 4. Calcular beta dinámico ETH→SOL en el lag óptimo
        self._beta_eth_sol = self._calcular_beta_lagged()
        
        # 5. Volatilidad para sizing
        sol_prices = np.array([self._sol_history[i] for i in range(min(60, self._sol_history.count))])[::-1]
        if len(sol_prices) > 1:
            retornos = np.diff(sol_prices) / sol_prices[:-1]
            self._volatilidad_reciente = np.std(retornos) * np.sqrt(60 * 24)
        else:
            self._volatilidad_reciente = 2.0
        
        # Filtro 3: Sizing Dinámico basado en Volatilidad
        # Volatilidad normal crypto: ~1.5, alta: >2.5, muy alta: >4.0
        if self._volatilidad_reciente > 4.0:
            size_factor = 0.3  # Solo 30% exposición en alta volatilidad
        elif self._volatilidad_reciente > 2.5:
            size_factor = 0.6  # 60% en volatilidad media-alta
        else:
            size_factor = 0.95  # 95% en volatilidad normal
        
        # Plotear métricas (validar NaN)
        if not np.isnan(partial_corr) and not np.isinf(partial_corr):
            self.plot("Correlacion", "Parcial ETH-SOL|BTC", partial_corr)
        
        self.plot("Lag", "Optimo (min)", float(self._lag_optimo))
        
        if not np.isnan(self._ccf_max) and not np.isinf(self._ccf_max):
            self.plot("Lag", "CCF Max", self._ccf_max)
        
        if not np.isnan(eth_lagged_signal) and not np.isinf(eth_lagged_signal):
            self.plot("Signal", "ETH Lagged", eth_lagged_signal * 100)
        
        if not np.isnan(self._beta_eth_sol) and not np.isinf(self._beta_eth_sol):
            self.plot("Beta", "ETH→SOL @Lag", self._beta_eth_sol)
        
        self.plot("Risk", "Volatilidad", self._volatilidad_reciente)
        self.plot("Risk", "Size Factor", size_factor)
        
        # Lógica de trading: ETH(t-lag) predice SOL(t)
        señal = self._evaluar_señal_con_lag(partial_corr, eth_lagged_signal)
        
        if señal == "LONG":
            self.set_holdings(self._sol, size_factor)
            self.debug(f"LONG SOL {size_factor:.0%} - ETH señal hace {self._lag_optimo}min, CCF: {self._ccf_max:.3f}")
        
        elif señal == "FLAT":
            if self.portfolio.invested:
                self.liquidate()
                self.debug(f"FLAT - Señal lagged débil")
    
    def _calcular_senal_limpia(self):
        """
        Calcula la correlación parcial entre ETH y SOL controlando por BTC.
        Retorna: (partial_correlation, residuo_eth_reciente, residuo_sol_reciente)
        """
        # Convertir ventanas a arrays
        eth_prices = np.array([self._eth_history[i] for i in range(self._eth_history.count)])[::-1]
        sol_prices = np.array([self._sol_history[i] for i in range(self._sol_history.count)])[::-1]
        btc_prices = np.array([self._btc_history[i] for i in range(self._btc_history.count)])[::-1]
        
        # Calcular retornos
        r_eth = np.diff(eth_prices) / eth_prices[:-1]
        r_sol = np.diff(sol_prices) / sol_prices[:-1]
        r_btc = np.diff(btc_prices) / btc_prices[:-1]
        
        # Verificar que hay variación en BTC
        if np.std(r_btc) < 1e-8:
            residuos_eth = r_eth
            residuos_sol = r_sol
        else:
            # Regresión 1: ETH = alpha + beta * BTC + error
            slope_eth, intercept_eth, _, _, _ = stats.linregress(r_btc, r_eth)
            prediccion_eth = intercept_eth + slope_eth * r_btc
            residuos_eth = r_eth - prediccion_eth
            
            # Regresión 2: SOL = alpha + beta * BTC + error
            slope_sol, intercept_sol, _, _, _ = stats.linregress(r_btc, r_sol)
            prediccion_sol = intercept_sol + slope_sol * r_btc
            residuos_sol = r_sol - prediccion_sol
        
        # Correlación Parcial
        if len(residuos_eth) > 1 and len(residuos_sol) > 1 and np.std(residuos_eth) > 1e-8 and np.std(residuos_sol) > 1e-8:
            partial_corr, _ = stats.pearsonr(residuos_eth, residuos_sol)
        else:
            partial_corr = 0
        
        return partial_corr, residuos_eth[-1], residuos_sol[-1]
    
    def _calcular_lag_optimo(self):
        """
        Calcula Cross-Correlation Function entre ETH y SOL para encontrar lag óptimo.
        Retorna: (lag_optimo, ccf_max, ccf_array)
        """
        eth_prices = np.array([self._eth_ccf[i] for i in range(self._eth_ccf.count)])[::-1]
        sol_prices = np.array([self._sol_ccf[i] for i in range(self._sol_ccf.count)])[::-1]
        
        # Calcular retornos
        r_eth = np.diff(eth_prices) / eth_prices[:-1]
        r_sol = np.diff(sol_prices) / sol_prices[:-1]
        
        # Normalizar
        r_eth = (r_eth - np.mean(r_eth)) / (np.std(r_eth) + 1e-8)
        r_sol = (r_sol - np.mean(r_sol)) / (np.std(r_sol) + 1e-8)
        
        # Calcular CCF para diferentes lags: ETH(t-k) vs SOL(t)
        ccf_values = []
        for lag in range(1, self._max_lag + 1):
            if len(r_eth) > lag and len(r_sol) > lag:
                # ETH lagged vs SOL actual
                eth_lagged = r_eth[:-lag]
                sol_actual = r_sol[lag:]
                
                # Validar que tienen varianza
                if len(eth_lagged) > 1 and len(sol_actual) > 1 and np.std(eth_lagged) > 1e-8 and np.std(sol_actual) > 1e-8:
                    corr = np.corrcoef(eth_lagged, sol_actual)[0, 1]
                    if not np.isnan(corr) and not np.isinf(corr):
                        ccf_values.append(corr)
                    else:
                        ccf_values.append(0)
                else:
                    ccf_values.append(0)
            else:
                ccf_values.append(0)
        
        # Encontrar lag con mayor correlación
        if ccf_values and any(v != 0 for v in ccf_values):
            max_idx = np.argmax(np.abs(ccf_values))
            lag_optimo = max_idx + 1
            ccf_max = ccf_values[max_idx]
        else:
            lag_optimo = 5
            ccf_max = 0
        
        return lag_optimo, ccf_max, ccf_values
    
    def _detectar_señal_lagged(self):
        """Detecta si ETH tuvo movimiento fuerte hace 'lag_optimo' minutos"""
        eth_prices = np.array([self._eth_ccf[i] for i in range(min(self._lag_optimo + 5, self._eth_ccf.count))])[::-1]
        
        if len(eth_prices) < self._lag_optimo + 2:
            return 0
        
        # Retorno de ETH hace 'lag' minutos
        precio_lag = eth_prices[self._lag_optimo]
        precio_antes = eth_prices[self._lag_optimo + 1] if self._lag_optimo + 1 < len(eth_prices) else precio_lag
        
        if precio_antes == 0:
            return 0
        
        retorno_lagged = (precio_lag - precio_antes) / precio_antes
        return retorno_lagged
    
    def _calcular_beta_lagged(self):
        """Calcula beta ETH(t-lag) → SOL(t)"""
        eth_prices = np.array([self._eth_ccf[i] for i in range(self._eth_ccf.count)])[::-1]
        sol_prices = np.array([self._sol_ccf[i] for i in range(self._sol_ccf.count)])[::-1]
        
        lag = self._lag_optimo
        
        if len(eth_prices) < lag + 10 or len(sol_prices) < lag + 10:
            return 1.5
        
        # Retornos lagged
        r_eth_lagged = np.diff(eth_prices[:-lag]) / eth_prices[:-lag-1]
        r_sol_actual = np.diff(sol_prices[lag:]) / sol_prices[lag:-1]
        
        # Asegurar misma longitud
        min_len = min(len(r_eth_lagged), len(r_sol_actual))
        r_eth_lagged = r_eth_lagged[:min_len]
        r_sol_actual = r_sol_actual[:min_len]
        
        if len(r_eth_lagged) < 2 or np.std(r_eth_lagged) < 1e-8:
            return 1.5
        
        # Beta: cov(SOL(t), ETH(t-lag)) / var(ETH(t-lag))
        beta = np.cov(r_sol_actual, r_eth_lagged)[0, 1] / np.var(r_eth_lagged)
        return max(0.5, min(3.0, beta))
    
    def _evaluar_señal_con_lag(self, partial_corr, eth_lagged_signal):
        """
        Evalúa señal usando lag explícito: ETH(t-lag) fuerte → SOL(t) debería seguir
        """
        # 1. ETH tuvo movimiento fuerte hace 'lag' minutos
        eth_señal_fuerte = eth_lagged_signal > 0.003  # >0.3%
        
        # 2. Alta CCF en el lag óptimo (estructura lead-lag confirmada)
        ccf_significativo = abs(self._ccf_max) > 0.3
        
        # 3. Alta correlación parcial (estructura base)
        alta_correlacion = partial_corr > self._umbral_parcial
        
        # 4. BTC neutral
        btc_prices = np.array([self._btc_history[i] for i in range(min(30, self._btc_history.count))])[::-1]
        if len(btc_prices) > 1:
            btc_momentum = (btc_prices[-1] - btc_prices[0]) / btc_prices[0]
            btc_no_domina = abs(btc_momentum) < 0.01
        else:
            btc_no_domina = True
        
        # Señal: ETH(t-lag) fuerte + CCF significativo + Correlación alta + BTC neutral
        if eth_señal_fuerte and ccf_significativo and alta_correlacion and btc_no_domina:
            return "LONG"
        
        # Exit si CCF se debilita o señal lagged se invierte
        if self.portfolio.invested:
            if abs(self._ccf_max) < 0.2 or eth_lagged_signal < -0.005:
                return "FLAT"
        
        return "HOLD"
    
    def _setup_charts(self):
        """Configura gráficos para análisis"""
        corr_chart = Chart("Correlacion")
        corr_chart.add_series(Series("Parcial ETH-SOL|BTC", SeriesType.LINE, "$", Color.BLUE))
        corr_chart.add_series(Series("Umbral", SeriesType.LINE, "$", Color.RED))
        self.add_chart(corr_chart)
        
        residuo_chart = Chart("Residuos")
        residuo_chart.add_series(Series("ETH Puro", SeriesType.LINE, "%", Color.GREEN))
        residuo_chart.add_series(Series("SOL Puro", SeriesType.LINE, "%", Color.ORANGE))
        self.add_chart(residuo_chart)
        
        macro_chart = Chart("Macro")
        macro_chart.add_series(Series("BTC Price", SeriesType.LINE, "$", Color.ORANGE))
        macro_chart.add_series(Series("SMA 200", SeriesType.LINE, "$", Color.RED))
        self.add_chart(macro_chart)
        
        risk_chart = Chart("Risk")
        risk_chart.add_series(Series("Drawdown %", SeriesType.LINE, "%", Color.RED))
        risk_chart.add_series(Series("Volatilidad", SeriesType.LINE, "$", Color.PURPLE))
        risk_chart.add_series(Series("Size Factor", SeriesType.LINE, "$", Color.GREEN))
        self.add_chart(risk_chart)
        
        # Plotear umbral fijo
        self.plot("Correlacion", "Umbral", self._umbral_parcial)
