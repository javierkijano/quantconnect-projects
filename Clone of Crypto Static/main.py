from AlgorithmImports import *
import numpy as np
from scipy import stats

class TrianguloDeControl(QCAlgorithm):
    
    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_cash(100000)
        
        # El Triángulo: ETH (Señal), SOL (Objetivo), BTC (Control)
        self._eth = self.add_crypto("ETHUSD", Resolution.HOUR).symbol
        self._sol = self.add_crypto("SOLUSD", Resolution.HOUR).symbol
        self._btc = self.add_crypto("BTCUSD", Resolution.HOUR).symbol
        
        # Configuración de ventanas
        self._ventana_regresion = 48  # 48 horas para calcular residuos (más reactivo)
        self._ventana_corta = 10      # 10 horas para detectar movimiento reciente
        self._umbral_parcial = 0.5    # Correlación parcial mínima (más permisivo)
        self._umbral_eth_puro = 0.001 # 0.1% movimiento puro de ETH (más sensible)
        
        # Históricos de precios
        self._eth_history = RollingWindow[float](self._ventana_regresion)
        self._sol_history = RollingWindow[float](self._ventana_regresion)
        self._btc_history = RollingWindow[float](self._ventana_regresion)
        
        # Warm up para llenar ventanas
        self.set_warm_up(self._ventana_regresion, Resolution.HOUR)
        
        # Indicadores para tracking
        self._partial_corr = 0
        self._residuo_eth = 0
        self._residuo_sol = 0
        
        # Rebalanceo cada hora
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.every(timedelta(hours=1)),
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
        if data.contains_key(self._sol) and data[self._sol] is not None:
            self._sol_history.add(float(data[self._sol].close))
        if data.contains_key(self._btc) and data[self._btc] is not None:
            self._btc_history.add(float(data[self._btc].close))
    
    def _rebalance(self):
        # Verificar que tenemos datos suficientes
        if not self._eth_history.is_ready or not self._sol_history.is_ready or not self._btc_history.is_ready:
            return
        
        # Calcular señal limpia
        partial_corr, residuo_eth, residuo_sol = self._calcular_senal_limpia()
        
        # Guardar para análisis
        self._partial_corr = partial_corr
        self._residuo_eth = residuo_eth
        self._residuo_sol = residuo_sol
        
        # Plotear métricas
        self.plot("Correlacion", "Parcial ETH-SOL|BTC", partial_corr)
        self.plot("Residuos", "ETH Puro", residuo_eth * 100)  # En porcentaje
        self.plot("Residuos", "SOL Puro", residuo_sol * 100)
        
        # Lógica de trading
        señal = self._evaluar_señal(partial_corr, residuo_eth)
        
        if señal == "LONG":
            # ETH muestra fuerza propia + alta correlación parcial → Comprar SOL
            self.set_holdings(self._sol, 0.95)
            self.debug(f"LONG SOL - Partial Corr: {partial_corr:.3f}, ETH Residuo: {residuo_eth:.4f}")
        
        elif señal == "FLAT":
            # Sin señal clara → Liquidar posiciones
            if self.portfolio[self._sol].invested:
                self.liquidate(self._sol)
                self.debug(f"FLAT - Partial Corr: {partial_corr:.3f}, ETH Residuo: {residuo_eth:.4f}")
    
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
        
        # Verificar que hay variación en BTC (necesario para regresión)
        if np.std(r_btc) < 1e-8:
            # Si BTC no se mueve, los residuos son los retornos completos
            residuos_eth = r_eth
            residuos_sol = r_sol
        else:
            # Regresión 1: ETH = alpha + beta * BTC + error
            slope_eth, intercept_eth, _, _, _ = stats.linregress(r_btc, r_eth)
            prediccion_eth = intercept_eth + slope_eth * r_btc
            residuos_eth = r_eth - prediccion_eth  # La "fuerza pura" de ETH
            
            # Regresión 2: SOL = alpha + beta * BTC + error
            slope_sol, intercept_sol, _, _, _ = stats.linregress(r_btc, r_sol)
            prediccion_sol = intercept_sol + slope_sol * r_btc
            residuos_sol = r_sol - prediccion_sol  # La "fuerza pura" de SOL
        
        # Correlación Parcial: Correlación de los residuos
        if len(residuos_eth) > 1 and len(residuos_sol) > 1 and np.std(residuos_eth) > 1e-8 and np.std(residuos_sol) > 1e-8:
            partial_corr, _ = stats.pearsonr(residuos_eth, residuos_sol)
        else:
            partial_corr = 0
        
        # Retornar correlación parcial y residuos recientes
        return partial_corr, residuos_eth[-1], residuos_sol[-1]
    
    def _evaluar_señal(self, partial_corr, residuo_eth):
        """
        Evalúa si hay señal de entrada basada en:
        1. ETH muestra fuerza propia (residuo positivo significativo)
        2. Alta correlación parcial con SOL
        """
        # Condición A: ETH sube por méritos propios (no por BTC)
        eth_muestra_fuerza = residuo_eth > self._umbral_eth_puro
        
        # Condición B: Alta correlación parcial histórica
        alta_correlacion = partial_corr > self._umbral_parcial
        
        # Verificar que BTC no está en tendencia fuerte (esto es un proxy, idealmente calcularíamos su volatilidad)
        btc_prices = np.array([self._btc_history[i] for i in range(min(10, self._btc_history.count))])[::-1]
        if len(btc_prices) > 1:
            btc_retorno_reciente = (btc_prices[-1] - btc_prices[0]) / btc_prices[0]
            btc_no_domina = abs(btc_retorno_reciente) < 0.05  # BTC no se movió más de 5% (más permisivo)
        else:
            btc_no_domina = True
        
        # Señal de compra: ETH fuerte + Correlación alta + BTC neutral
        if eth_muestra_fuerza and alta_correlacion and btc_no_domina:
            return "LONG"
        
        # Si estamos en posición pero las condiciones ya no se cumplen
        if self.portfolio[self._sol].invested:
            # Exit si la correlación cae mucho o ETH pierde fuerza significativa
            if partial_corr < 0.3 or residuo_eth < -0.002:
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
        
        # Plotear umbral fijo
        self.plot("Correlacion", "Umbral", self._umbral_parcial)
