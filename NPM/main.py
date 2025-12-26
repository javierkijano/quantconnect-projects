from AlgorithmImports import *
import numpy as np
from collections import deque
import random


class ResidualPolicyEnsembleQC(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(100000)

        # --- Universe ---
        self._symbol = self.add_equity("SPY", Resolution.DAILY).symbol

        # --- PARAMETROS MEJORADOS ---
        self.lookback = 150  # mas datos para mejor entrenamiento
        self.rebalance_days = 3  # rebalanceo mas frecuente
        self.selection_days = 14  # seleccion cada 2 semanas
        self.target_leverage = 2.0  # leverage aumentado
        
        # Training config mejorado
        self.train_on_init = True
        self.training_epochs = 150  # mas epocas
        self.learning_rate = 0.003  # learning rate mas conservador
        self.training_data_collected = False

        # Ensemble config optimizado
        self.N = 300  # mas politicas candidatas
        self.k = 24  # mas direcciones en subespacio
        self.topX = 30  # mas politicas en ensemble final
        self.sigma = 0.12  # mayor exploracion
        self.beta = 0.35  # balance diversidad/score
        self.turnover_penalty = 0.10  # menor penalizacion turnover

        # Trading mas activo
        self.trade_threshold = 0.005  # mas sensible a cambios

        # --- Data buffers ---
        self.prices = deque(maxlen=self.lookback + 10)
        self.returns = deque(maxlen=self.lookback + 10)
        self.volumes = deque(maxlen=self.lookback + 10)  # nuevo: volumen
        self.training_data = []

        # Per-policy tracking
        self.policy_returns = {i: deque(maxlen=252) for i in range(self.N)}
        self.policy_prev_p = {i: 0.0 for i in range(self.N)}

        # Selected policies
        self.selected = []
        self.weights = {}

        # Schedule
        self.last_rebalance = None
        self.last_selection = None

        # --- Red neuronal mas grande ---
        self._d = self._feature_dim()  # 12 features ahora
        self.hidden1 = 48  # capa mas grande
        self.hidden2 = 24  # capa mas grande

        # Base weights
        random.seed(42)  # seed diferente
        self.w0 = self._init_base_weights()
        self.w0_trained = False

        # Direcciones para perturbaciones
        self.u = self._make_last_layer_directions(self.k)

        # Alphas para N politicas
        alpha_list = []
        for i in range(self.N):
            row = [random.gauss(0, self.sigma) for _ in range(self.k)]
            alpha_list.append(row)
        self._alpha = np.array(alpha_list)

        # Warmup
        self.set_warm_up(self.lookback + 30)

        self.debug(f"IMPROVED: {self.N} policies, {self.k} dirs, top {self.topX}")
        self.debug(f"Features: {self._d}, Hidden: {self.hidden1}-{self.hidden2}")
        self.debug(f"Leverage: {self.target_leverage}x, Rebalance: {self.rebalance_days}d")

    def on_data(self, data: Slice):
        if self.is_warming_up:
            if self._symbol in data and data[self._symbol] is not None:
                self._update_buffers(data)
                if self.train_on_init and len(self.returns) >= 40:
                    self._collect_training_sample()
            return
        
        # Train despues de warmup
        if self.train_on_init and not self.w0_trained and not self.is_warming_up:
            self._train_base_model()
            self.w0_trained = True

        if self._symbol not in data or data[self._symbol] is None:
            return

        self._update_buffers(data)
        if len(self.returns) < 40:
            return

        t = self.time

        # Seleccion periodica
        if self.last_selection is None or (t - self.last_selection).days >= self.selection_days:
            self._select_policies()
            self.last_selection = t

        # Rebalanceo periodico
        if self.last_rebalance is None or (t - self.last_rebalance).days >= self.rebalance_days:
            self._rebalance()
            self.last_rebalance = t

    def _select_policies(self):
        min_hist = 25
        valid = []
        scores = {}
        
        for i in range(self.N):
            r = np.array(self.policy_returns[i], dtype=float)
            if r.size < min_hist:
                continue

            mu = r.mean()
            sd = r.std() + 1e-12
            sharpe = np.sqrt(252) * mu / sd
            
            # Sortino ratio como bonus
            downside = r[r < 0]
            downside_std = downside.std() + 1e-12 if len(downside) > 5 else sd
            sortino = np.sqrt(252) * mu / downside_std
            
            # Score combinado: Sharpe + Sortino bonus - turnover
            turnover_proxy = min(1.0, sd * 8.0)
            score = sharpe + 0.3 * sortino - self.turnover_penalty * turnover_proxy

            scores[i] = score
            valid.append(i)

        if len(valid) < max(5, self.topX // 2):
            return

        # Top M por score
        M = min(80, len(valid))
        topM = sorted(valid, key=lambda i: scores[i], reverse=True)[:M]

        # Matriz de correlacion
        R = [np.array(self.policy_returns[i], dtype=float) for i in topM]
        L = min(len(x) for x in R)
        R_stacked = np.stack([x[-L:] for x in R], axis=0)
        noise = np.array([[random.gauss(0, 1e-10) for _ in range(L)] for _ in range(len(topM))])
        R_stacked = R_stacked + noise
        corr = np.corrcoef(R_stacked)

        # Seleccion greedy con diversidad
        chosen_idx = []
        best = max(range(len(topM)), key=lambda j: scores[topM[j]])
        chosen_idx.append(best)

        while len(chosen_idx) < self.topX and len(chosen_idx) < len(topM):
            best_j = None
            best_val = -1e18
            for j in range(len(topM)):
                if j in chosen_idx:
                    continue
                mean_corr = float(np.mean([abs(corr[j, c]) for c in chosen_idx]))
                val = scores[topM[j]] - self.beta * mean_corr
                if val > best_val:
                    best_val = val
                    best_j = j
            if best_j is not None:
                chosen_idx.append(best_j)

        self.selected = [topM[j] for j in chosen_idx]
        
        # Pesos proporcionales al score (no iguales)
        sel_scores = [max(0.01, scores[i] + 5) for i in self.selected]  # shift para positivos
        total_score = sum(sel_scores)
        self.weights = {self.selected[j]: sel_scores[j] / total_score for j in range(len(self.selected))}

        avg_score = sum(scores[i] for i in self.selected) / len(self.selected)
        self.debug(f"Selected {len(self.selected)} policies, avg score: {avg_score:.3f}")

    def _rebalance(self):
        if not self.selected:
            self.set_holdings(self._symbol, 0.0)
            return

        s = self._compute_state()
        
        # Senal agregada con pesos
        p_total = 0.0
        for i in self.selected:
            p_i = self._policy_output(i, s)
            p_total += self.weights[i] * p_i

        # Aplicar leverage y clipping
        p_total = float(np.clip(p_total, -1.0, 1.0)) * self.target_leverage

        # Throttle
        current = self.portfolio[self._symbol].holdings_value / max(1.0, self.portfolio.total_portfolio_value)
        change = abs(p_total - current)
        
        if change < self.trade_threshold:
            return

        self.set_holdings(self._symbol, p_total)

    def _policy_output(self, i: int, s: np.ndarray) -> float:
        z0 = self._forward_logit(s, self.w0)
        w1 = self._apply_delta_to_last_layer(self.w0, self._alpha[i])
        z1 = self._forward_logit(s, w1)
        z_res = z1 - z0
        return float(np.tanh(z_res))

    def _forward_logit(self, s: np.ndarray, w: dict) -> float:
        x = s
        h1 = np.maximum(0.0, x @ w["W1"] + w["b1"])
        h2 = np.maximum(0.0, h1 @ w["W2"] + w["b2"])
        z = float(h2 @ w["W3"] + w["b3"])
        return z

    def _apply_delta_to_last_layer(self, w0: dict, alpha_i) -> dict:
        delta_vec = np.zeros(self.hidden2 + 1, dtype=float)
        for j in range(self.k):
            delta_vec += alpha_i[j] * self.u[j]

        w = {
            "W1": w0["W1"].copy(), "b1": w0["b1"].copy(),
            "W2": w0["W2"].copy(), "b2": w0["b2"].copy(),
            "W3": w0["W3"].copy(), "b3": float(w0["b3"]),
        }
        w["W3"] = (w["W3"].reshape(-1) + delta_vec[:self.hidden2]).reshape(self.hidden2, 1)
        w["b3"] = float(w["b3"] + delta_vec[-1])
        return w

    def _update_buffers(self, data: Slice):
        bar = data[self._symbol]
        price = float(bar.Close)
        volume = float(bar.Volume) if bar.Volume else 0
        
        if self.prices:
            prev = self.prices[-1]
            r = np.log(price / prev)
            self.returns.append(r)
        self.prices.append(price)
        self.volumes.append(volume)
    
    def _collect_training_sample(self):
        if len(self.returns) < 40:
            return
        
        s = self._compute_state()
        if len(self.returns) >= 5:
            fwd_return = sum(list(self.returns)[-5:])
            self.training_data.append((s.copy(), fwd_return))

        # Actualizar returns de politicas
        if len(self.returns) >= 2:
            last_r = self.returns[-1]
            for i in range(self.N):
                p = self._policy_output(i, s)
                pnl = p * last_r
                dp = abs(p - self.policy_prev_p[i])
                pnl_adj = pnl - 0.0003 * dp  # menor costo
                self.policy_prev_p[i] = p
                self.policy_returns[i].append(pnl_adj)

    def _compute_state(self) -> np.ndarray:
        prices = np.array(self.prices, dtype=float)
        rets = np.array(self.returns, dtype=float)
        vols = np.array(self.volumes, dtype=float)

        # Returns a diferentes horizontes
        r1 = rets[-1] if len(rets) >= 1 else 0
        r5 = rets[-5:].sum() if len(rets) >= 5 else rets.sum()
        r10 = rets[-10:].sum() if len(rets) >= 10 else rets.sum()
        r20 = rets[-20:].sum() if len(rets) >= 20 else rets.sum()

        # Volatilidad
        vol10 = rets[-10:].std() if len(rets) >= 10 else rets.std() + 1e-8
        vol20 = rets[-20:].std() if len(rets) >= 20 else rets.std() + 1e-8
        vol60 = rets[-60:].std() if len(rets) >= 60 else rets.std() + 1e-8

        # EMAs
        ema20 = self._ema(prices, 20)
        ema50 = self._ema(prices, 50)
        ema200 = self._ema(prices, 200)
        p = prices[-1]
        
        dist_ema20 = (p - ema20) / (ema20 + 1e-12)
        dist_ema200 = (p - ema200) / (ema200 + 1e-12)

        # Momentum slope
        ema20_series = self._ema_series(prices, 20)
        slope = (ema20_series[-1] - ema20_series[-10]) / (abs(ema20_series[-10]) + 1e-12) if len(ema20_series) >= 10 else 0

        # Volume ratio
        vol_ma = vols[-20:].mean() if len(vols) >= 20 else vols.mean() + 1e-8
        vol_ratio = (vols[-1] / vol_ma - 1) if vol_ma > 0 else 0

        # Normalizar
        scale = vol20 + 1e-6
        feats = np.array([
            r1 / scale,
            r5 / scale,
            r10 / scale,
            r20 / scale,
            vol10 * np.sqrt(252),
            vol20 * np.sqrt(252),
            vol60 * np.sqrt(252),
            slope * 10,
            dist_ema20 * 10,
            dist_ema200 * 10,
            vol_ratio,
            (ema20 - ema50) / (ema50 + 1e-12) * 10,  # trend strength
        ], dtype=float)

        feats = np.clip(feats, -10, 10)
        return feats

    def _feature_dim(self) -> int:
        return 12  # 12 features ahora
    
    def _train_base_model(self):
        if len(self.training_data) < 60:
            self.debug(f"Insufficient training data: {len(self.training_data)}")
            return
        
        self.debug(f"Training on {len(self.training_data)} samples, {self.training_epochs} epochs")
        
        X = np.array([s for s, _ in self.training_data])
        y = np.array([r for _, r in self.training_data])
        
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_norm = np.clip((y - y_mean) / y_std, -3, 3)
        
        best_loss = float('inf')
        patience = 15
        no_improve = 0
        
        for epoch in range(self.training_epochs):
            total_loss = 0.0
            indices = list(range(len(X)))
            random.shuffle(indices)
            
            for idx in indices:
                s = X[idx]
                target = y_norm[idx]
                
                z, h1, h2 = self._forward_with_cache(s, self.w0)
                pred = np.tanh(z)
                error = pred - target
                loss = 0.5 * error * error
                total_loss += loss
                
                d_z = error * (1.0 - pred * pred)
                d_W3 = h2.reshape(-1, 1) * d_z
                d_b3 = d_z
                d_h2 = self.w0["W3"].flatten() * d_z
                d_h2 = d_h2 * (h2 > 0)
                d_W2 = np.outer(h1, d_h2)
                d_b2 = d_h2
                d_h1 = self.w0["W2"] @ d_h2
                d_h1 = d_h1 * (h1 > 0)
                d_W1 = np.outer(s, d_h1)
                d_b1 = d_h1
                
                lr = self.learning_rate
                self.w0["W3"] -= lr * np.clip(d_W3, -1, 1)
                self.w0["b3"] -= lr * np.clip(d_b3, -1, 1)
                self.w0["W2"] -= lr * np.clip(d_W2, -1, 1)
                self.w0["b2"] -= lr * np.clip(d_b2, -1, 1)
                self.w0["W1"] -= lr * np.clip(d_W1, -1, 1)
                self.w0["b1"] -= lr * np.clip(d_b1, -1, 1)
            
            avg_loss = total_loss / len(X)
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 20 == 0:
                self.debug(f"Epoch {epoch}: loss = {avg_loss:.6f}")
            
            if no_improve >= patience:
                self.debug(f"Early stopping at epoch {epoch}")
                break
        
        self.debug(f"Training complete. Final loss: {best_loss:.6f}")
        self.training_data = []
    
    def _forward_with_cache(self, s: np.ndarray, w: dict):
        h1 = np.maximum(0.0, s @ w["W1"] + w["b1"])
        h2 = np.maximum(0.0, h1 @ w["W2"] + w["b2"])
        z = float(h2 @ w["W3"] + w["b3"])
        return z, h1, h2

    def _ema(self, x: np.ndarray, period: int) -> float:
        if x.size < 2:
            return float(x[-1]) if x.size else 0.0
        alpha = 2.0 / (period + 1.0)
        ema = x[0]
        for v in x[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return float(ema)

    def _ema_series(self, x: np.ndarray, period: int):
        if x.size == 0:
            return []
        alpha = 2.0 / (period + 1.0)
        ema = x[0]
        out = [float(ema)]
        for v in x[1:]:
            ema = alpha * v + (1 - alpha) * ema
            out.append(float(ema))
        return out

    def _init_base_weights(self):
        W1 = np.array([[random.gauss(0, 0.015) for _ in range(self.hidden1)] for _ in range(self._d)])
        b1 = np.zeros(self.hidden1)
        W2 = np.array([[random.gauss(0, 0.015) for _ in range(self.hidden2)] for _ in range(self.hidden1)])
        b2 = np.zeros(self.hidden2)
        W3 = np.array([[random.gauss(0, 0.015)] for _ in range(self.hidden2)])
        b3 = 0.0
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def _make_last_layer_directions(self, k):
        dim = self.hidden2 + 1
        U = []
        for _ in range(k):
            v = np.array([random.gauss(0, 1) for _ in range(dim)])
            v = v / (np.linalg.norm(v) + 1e-12)
            U.append(v)
        return U
