# region imports
from AlgorithmImports import *
# endregion

import math
from datetime import datetime, timedelta

import numpy as np


class ResidualStateEngineAlgorithm(QCAlgorithm):
    """Residual state engine with PCA residuals and KKT optimization."""

    def Initialize(self):
        self._load_parameters()

        self.SetStartDate(
            self.param_start_date.year,
            self.param_start_date.month,
            self.param_start_date.day,
        )
        self.SetEndDate(self.param_end_date.year, self.param_end_date.month, self.param_end_date.day)
        self.SetCash(100000)

        self.symbols = []
        for ticker in self.tickers:
            if self.asset_type == "forex":
                security = self.AddForex(ticker, self.data_resolution, Market.Oanda)
            else:
                security = self.AddEquity(ticker, self.data_resolution)
                security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
            self.symbols.append(security.Symbol)

        if len(self.symbols) < 2:
            self.Debug("Need at least two symbols for the residual system.")
            return

        self.N = len(self.symbols)
        self.K = min(self.K, self.N - 1)
        if self.K < 1:
            self.K = 1

        self.return_windows = {
            symbol: RollingWindow[float](self.lookback_returns) for symbol in self.symbols
        }
        self.prev_prices = {symbol: None for symbol in self.symbols}

        self.persist_counter = 0
        self.shutdown_until = None
        self.prev_P = None
        self.last_weights = np.zeros(self.N)

        self.rebalance_counter = 0
        self.rebalance_count = 0
        self.trade_count = 0
        self.shutdown_count = 0
        self.terminated_due_to_failure = False

        self.metric_w_norm = []
        self.metric_energy = []
        self.metric_turnover = []
        self.metric_neutrality = []
        self.metric_alignment = []
        self.metric_subspace_angle = []
        self.metric_factor_leakage = []

        chart = Chart("Health")
        chart.AddSeries(Series("w_norm", SeriesType.Line, 0))
        chart.AddSeries(Series("energy", SeriesType.Line, 0))
        chart.AddSeries(Series("turnover", SeriesType.Line, 0))
        chart.AddSeries(Series("neutrality_err", SeriesType.Line, 0))
        chart.AddSeries(Series("signal_strength", SeriesType.Line, 0))
        chart.AddSeries(Series("alignment", SeriesType.Line, 0))
        chart.AddSeries(Series("subspace_angle", SeriesType.Line, 0))
        chart.AddSeries(Series("factor_leakage", SeriesType.Line, 0))
        self.AddChart(chart)

        self.SetWarmUp(self.lookback_returns + 60, self.data_resolution)

        if self.asset_type == "forex" and self.data_resolution == Resolution.Hour:
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.Every(timedelta(hours=1)),
                self.Rebalance,
            )
        elif self.asset_type == "forex":
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.At(0, 1),
                self.Rebalance,
            )
        else:
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.AfterMarketOpen(self.symbols[0], 1),
                self.Rebalance,
            )

    def OnData(self, data: Slice):
        for symbol in self.symbols:
            bar = None
            if data.Bars.ContainsKey(symbol):
                bar = data.Bars[symbol]
            elif data.QuoteBars.ContainsKey(symbol):
                bar = data.QuoteBars[symbol]
            if bar is None:
                continue
            price = getattr(bar, "Close", None)
            if price is None and bar.Bid is not None and bar.Ask is not None:
                price = (bar.Bid.Close + bar.Ask.Close) / 2.0
            if price <= 0:
                continue
            prev = self.prev_prices.get(symbol)
            if prev is not None and prev > 0:
                ret = math.log(price / prev)
                self.return_windows[symbol].Add(ret)
            self.prev_prices[symbol] = price

    def Rebalance(self):
        self.rebalance_counter += 1
        if self.rebalance_counter % self.rebalance_days != 0:
            return

        self.rebalance_count += 1

        if self.IsWarmingUp:
            self._apply_weights(np.zeros(self.N))
            self.persist_counter = 0
            self._log_state(
                state="no_signal",
                w=np.zeros(self.N),
                strength=0.0,
                neutrality_err=0.0,
                energy=0.0,
                turnover=0.0,
                alignment=0.0,
                T=0,
            )
            self.last_weights = np.zeros(self.N)
            return

        if self.shutdown_until is not None and self.Time < self.shutdown_until:
            self._apply_weights(np.zeros(self.N))
            self.persist_counter = 0
            self._log_state(
                state="shutdown_cooldown",
                w=np.zeros(self.N),
                strength=0.0,
                neutrality_err=0.0,
                energy=0.0,
                turnover=0.0,
                alignment=0.0,
                T=0,
            )
            self.last_weights = np.zeros(self.N)
            return

        X = self._build_return_matrix()
        if X is None:
            self._apply_weights(np.zeros(self.N))
            self.persist_counter = 0
            self._log_state(
                state="no_signal",
                w=np.zeros(self.N),
                strength=0.0,
                neutrality_err=0.0,
                energy=0.0,
                turnover=0.0,
                alignment=0.0,
                T=0,
            )
            self.last_weights = np.zeros(self.N)
            return

        T = X.shape[0]

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
        std = np.where(std == 0, 1.0, std)
        Xn = (X - mean) / std
        if self.clip_sigma > 0:
            Xn = np.clip(Xn, -self.clip_sigma, self.clip_sigma)

        Xc = Xn - np.mean(Xn, axis=0)
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        except Exception:
            self._apply_weights(np.zeros(self.N))
            self.persist_counter = 0
            self._log_state(
                state="no_signal",
                w=np.zeros(self.N),
                strength=0.0,
                neutrality_err=0.0,
                energy=0.0,
                turnover=0.0,
                alignment=0.0,
                T=T,
            )
            self.last_weights = np.zeros(self.N)
            return

        P = Vt[: self.K, :].T

        r_t = Xn[-1]
        proj = P @ (P.T @ r_t)
        eps_t = r_t - proj
        r_norm = float(np.linalg.norm(r_t))
        proj_norm = float(np.linalg.norm(proj))
        factor_leakage = float(proj_norm / r_norm) if r_norm > 0 else 0.0
        angle = None
        subspace_angle = 0.0
        if self.prev_P is not None:
            angle = self._subspace_angle(self.prev_P, P)
            if angle is not None:
                subspace_angle = float(angle)

        E = Xc - (Xc @ P @ P.T)
        if E.shape[0] < 2:
            self._apply_weights(np.zeros(self.N))
            self.persist_counter = 0
            self._log_state(
                state="no_signal",
                w=np.zeros(self.N),
                strength=0.0,
                neutrality_err=0.0,
                energy=0.0,
                turnover=0.0,
                alignment=0.0,
                T=T,
                subspace_angle=subspace_angle,
                factor_leakage=factor_leakage,
            )
            self.last_weights = np.zeros(self.N)
            return

        S = np.cov(E, rowvar=False, bias=False)
        D = np.diag(np.diag(S))
        alpha = min(max(self.shrink_alpha, 0.0), 1.0)
        Sigma = (1.0 - alpha) * S + alpha * D
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma = Sigma + self.eps_diag * np.eye(self.N)

        invSig = None
        try:
            invSig = np.linalg.inv(Sigma + self.inv_delta * np.eye(self.N))
            val = float(eps_t @ invSig @ eps_t)
            strength = float(math.sqrt(max(val, 0.0)))
        except Exception:
            strength = float(np.linalg.norm(eps_t))

        if self.strategy_variant == "VARIANT_2_SIMPLE_SIGNAL":
            s = -eps_t
        else:
            s = -invSig @ eps_t if invSig is not None else -eps_t

        if strength >= self.signal_threshold:
            self.persist_counter += 1
        else:
            self.persist_counter = 0

        if not self._check_coherence(eps_t):
            self._apply_weights(np.zeros(self.N))
            self._log_state(
                state="no_coherence",
                w=np.zeros(self.N),
                strength=strength,
                neutrality_err=0.0,
                energy=0.0,
                turnover=float(np.sum(np.abs(self.last_weights))),
                alignment=0.0,
                T=T,
                subspace_angle=subspace_angle,
                factor_leakage=factor_leakage,
            )
            self.last_weights = np.zeros(self.N)
            self.prev_P = P
            return

        if self.persist_counter < self.signal_persist:
            self._apply_weights(np.zeros(self.N))
            self._log_state(
                state="no_signal",
                w=np.zeros(self.N),
                strength=strength,
                neutrality_err=0.0,
                energy=0.0,
                turnover=float(np.sum(np.abs(self.last_weights))),
                alignment=0.0,
                T=T,
                subspace_angle=subspace_angle,
                factor_leakage=factor_leakage,
            )
            self.last_weights = np.zeros(self.N)
            self.prev_P = P
            return

        R = self._build_regularization(Sigma)
        w_candidate = self._solve_kkt(R, P, s)
        if w_candidate is None:
            w_candidate = np.zeros(self.N)

        w_candidate = np.clip(w_candidate, -self.per_asset_cap, self.per_asset_cap)
        w_candidate = self._project_to_constraints(w_candidate, P)
        gross = float(np.sum(np.abs(w_candidate)))
        if gross > 0:
            max_abs = float(np.max(np.abs(w_candidate)))
            cap_scale = self.per_asset_cap / max_abs if max_abs > 0 else 1.0
            target_scale = self.gross_leverage / gross
            scale = min(target_scale, cap_scale)
            w_candidate = w_candidate * scale

        ones = np.ones(self.N)
        neutrality_err = float(
            max(abs(w_candidate @ ones), np.max(np.abs(w_candidate @ P)) if self.K > 0 else 0.0)
        )
        w_norm = float(np.linalg.norm(w_candidate))
        energy = float(w_candidate @ (R @ w_candidate)) if R is not None else 0.0
        turnover = float(np.sum(np.abs(w_candidate - self.last_weights)))
        alignment = self._alignment(w_candidate, s)

        state = "rebalance_ok"
        if angle is not None and angle > self.subspace_break_tol:
            state = "shutdown_subspace"

        if neutrality_err > self.neutrality_fail_tol:
            state = "shutdown_neutrality"

        if energy > self.max_energy:
            state = "shutdown_energy"

        if state.startswith("shutdown"):
            self._enter_shutdown()
            self._apply_weights(np.zeros(self.N))
            self._log_state(
                state=state,
                w=w_candidate,
                strength=strength,
                neutrality_err=neutrality_err,
                energy=energy,
                turnover=turnover,
                alignment=alignment,
                T=T,
                subspace_angle=subspace_angle,
                factor_leakage=factor_leakage,
            )
            self.last_weights = np.zeros(self.N)
            self.prev_P = P
            return

        self._apply_weights(w_candidate)
        if w_norm > 0:
            self.trade_count += 1

        self._log_state(
            state=state,
            w=w_candidate,
            strength=strength,
            neutrality_err=neutrality_err,
            energy=energy,
            turnover=turnover,
            alignment=alignment,
            T=T,
            subspace_angle=subspace_angle,
            factor_leakage=factor_leakage,
        )
        self.last_weights = w_candidate
        self.prev_P = P

    def OnEndOfAlgorithm(self):
        avg_w = float(np.mean(self.metric_w_norm)) if self.metric_w_norm else 0.0
        max_w = float(np.max(self.metric_w_norm)) if self.metric_w_norm else 0.0
        avg_energy = float(np.mean(self.metric_energy)) if self.metric_energy else 0.0
        max_energy = float(np.max(self.metric_energy)) if self.metric_energy else 0.0
        avg_turnover = float(np.mean(self.metric_turnover)) if self.metric_turnover else 0.0
        max_turnover = float(np.max(self.metric_turnover)) if self.metric_turnover else 0.0
        avg_neutrality = float(np.mean(self.metric_neutrality)) if self.metric_neutrality else 0.0
        max_neutrality = float(np.max(self.metric_neutrality)) if self.metric_neutrality else 0.0

        self.Debug(
            "SUMMARY | rebalances={0} trades={1} shutdowns={2} | "
            "avg_w_norm={3:.4f} max_w_norm={4:.4f} | "
            "avg_energy={5:.4f} max_energy={6:.4f} | "
            "avg_turnover={7:.4f} max_turnover={8:.4f} | "
            "avg_neutrality={9:.6f} max_neutrality={10:.6f}".format(
                self.rebalance_count,
                self.trade_count,
                self.shutdown_count,
                avg_w,
                max_w,
                avg_energy,
                max_energy,
                avg_turnover,
                max_turnover,
                avg_neutrality,
                max_neutrality,
            )
        )

        if self.terminated_due_to_failure:
            self.Debug("SYSTEM TERMINATED DUE TO STRUCTURAL FAILURE")
        else:
            self.Debug("SYSTEM COMPLETED WITHOUT STRUCTURAL VIOLATIONS")

    def _load_parameters(self):
        self.dataset_profile = self._get_param("dataset_profile", "PROFILE_A_CORE_ETFS").upper()
        self.strategy_variant = self._get_param("strategy_variant", "VARIANT_1_BASELINE").upper()
        self.asset_type = self._get_param("asset_type", "equity").lower()
        self.resolution_name = self._get_param("resolution", "daily").lower()
        self.data_resolution = self._parse_resolution(self.resolution_name)

        tickers_param = self._get_param("tickers", "")
        self.tickers = self._resolve_tickers(tickers_param, self.dataset_profile)

        self.lookback_returns = self._get_int_param("lookback_returns", 252)
        self.K = self._get_int_param("K", 3)
        self.rebalance_days = max(1, self._get_int_param("rebalance_days", 5))

        self.lambda_l2 = self._get_float_param("lambda_l2", 1e-2)
        self.lambda_cov = self._get_float_param("lambda_cov", 1e-1)
        self.lambda_fragile = self._get_float_param("lambda_fragile", 1e-2)

        self.shrink_alpha = self._get_float_param("shrink_alpha", 0.2)
        self.signal_threshold = self._get_float_param("signal_threshold", 2.0)
        self.signal_persist = self._get_int_param("signal_persist", 2)

        self.min_participation_frac = self._get_float_param("min_participation_frac", 0.4)
        self.participation_eps = self._get_float_param("participation_eps", 0.1)

        self.gross_leverage = self._get_float_param("gross_leverage", 1.0)
        self.per_asset_cap = self._get_float_param("per_asset_cap", 0.30)

        self.neutrality_tol = self._get_float_param("neutrality_tol", 1e-4)
        self.neutrality_fail_tol = self._get_float_param("neutrality_fail_tol", 1e-3)

        self.max_energy = self._get_float_param("max_energy", 10.0)
        self.subspace_break_tol = self._get_float_param("subspace_break_tol", 0.5)
        self.shutdown_cooldown_days = self._get_int_param("shutdown_cooldown_days", 10)

        self.clip_sigma = self._get_float_param("clip_sigma", 5.0)

        self.param_start_date = self._get_date_param("start_date", datetime(2013, 1, 1))
        self.param_end_date = self._get_date_param("end_date", datetime(2024, 1, 1))

        if self.strategy_variant not in {
            "VARIANT_1_BASELINE",
            "VARIANT_2_SIMPLE_SIGNAL",
            "VARIANT_3_NO_FRAGILE_PENALTY",
            "VARIANT_4_STRONG_OBJECT",
        }:
            self.strategy_variant = "VARIANT_1_BASELINE"

        self.eps_diag = 1e-6
        self.inv_delta = 1e-6

    def _resolve_tickers(self, tickers_param, dataset_profile):
        if tickers_param:
            tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
            return list(dict.fromkeys(tickers))

        profiles = {
            "PROFILE_A_CORE_ETFS": ["SPY", "QQQ", "IWM", "DIA", "EFA"],
            "PROFILE_B_SECTOR_ETFS": ["XLF", "XLK", "XLE", "XLV", "XLY", "XLP"],
            "PROFILE_C_BONDS_RATES": ["IEF", "TLT", "SHY", "LQD", "HYG"],
            "PROFILE_D_COMMODITIES_ETFS": ["GLD", "SLV", "USO", "DBA", "UNG"],
        }
        return profiles.get(dataset_profile, profiles["PROFILE_A_CORE_ETFS"])

    def _parse_resolution(self, resolution_name):
        mapping = {
            "daily": Resolution.Daily,
            "hour": Resolution.Hour,
            "minute": Resolution.Minute,
        }
        return mapping.get(resolution_name, Resolution.Daily)

    def _get_param(self, name, default):
        value = self.GetParameter(name)
        if value is None or value == "":
            return default
        return value

    def _get_int_param(self, name, default):
        value = self.GetParameter(name)
        if value is None or value == "":
            return default
        try:
            return int(float(value))
        except ValueError:
            return default

    def _get_float_param(self, name, default):
        value = self.GetParameter(name)
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _get_date_param(self, name, default):
        value = self.GetParameter(name)
        if value is None or value == "":
            return default
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return default

    def _build_return_matrix(self):
        T = self.lookback_returns
        X = np.zeros((T, self.N))
        for j, symbol in enumerate(self.symbols):
            window = self.return_windows[symbol]
            if window.Count < T:
                return None
            for i in range(T):
                X[T - 1 - i, j] = window[i]
        return X

    def _check_coherence(self, eps_t):
        counts = np.sum(np.abs(eps_t) > self.participation_eps)
        return counts >= int(math.ceil(self.min_participation_frac * self.N))

    def _build_regularization(self, Sigma):
        lambda_fragile = self.lambda_fragile
        if self.strategy_variant == "VARIANT_3_NO_FRAGILE_PENALTY":
            lambda_fragile = 0.0
        elif self.strategy_variant == "VARIANT_4_STRONG_OBJECT":
            lambda_fragile = 10.0 * lambda_fragile

        fragile_penalty = np.zeros_like(Sigma)
        if lambda_fragile > 0:
            try:
                fragile_penalty = np.linalg.inv(Sigma + self.inv_delta * np.eye(self.N))
            except Exception:
                diag = np.diag(Sigma)
                safe = np.maximum(diag, self.inv_delta)
                fragile_penalty = np.diag(1.0 / safe)

        R = (
            self.lambda_l2 * np.eye(self.N)
            + self.lambda_cov * Sigma
            + lambda_fragile * fragile_penalty
        )
        return 0.5 * (R + R.T)

    def _solve_kkt(self, R, P, s):
        if R is None or s is None:
            return None
        m = 1 + self.K
        A = np.zeros((m, self.N))
        A[0, :] = 1.0
        A[1:, :] = P.T
        top = np.hstack([2.0 * R, A.T])
        bottom = np.hstack([A, np.zeros((m, m))])
        kkt = np.vstack([top, bottom])
        rhs = np.concatenate([s, np.zeros(m)])
        try:
            sol = np.linalg.solve(kkt, rhs)
        except Exception:
            return None
        return sol[: self.N]

    def _project_to_constraints(self, w, P):
        if w is None:
            return None
        R = np.eye(self.N)
        projected = self._solve_kkt(R, P, 2.0 * w)
        return projected if projected is not None else w

    def _subspace_angle(self, P1, P2):
        try:
            Q1, _ = np.linalg.qr(P1)
            Q2, _ = np.linalg.qr(P2)
            _, svals, _ = np.linalg.svd(Q1.T @ Q2)
            svals = np.clip(svals, -1.0, 1.0)
            angles = np.arccos(svals)
            return float(np.max(angles))
        except Exception:
            return None

    def _enter_shutdown(self):
        self.shutdown_count += 1
        self.terminated_due_to_failure = True
        self.persist_counter = 0
        self.shutdown_until = self.Time + timedelta(days=self.shutdown_cooldown_days)

    def _apply_weights(self, weights):
        targets = [PortfolioTarget(symbol, float(weights[i])) for i, symbol in enumerate(self.symbols)]
        self.SetHoldings(targets)

    def _alignment(self, w, s):
        if s is None:
            return 0.0
        denom = float(np.linalg.norm(w) * np.linalg.norm(s))
        if denom == 0:
            return 0.0
        return float((w @ s) / denom)

    def _log_state(
        self,
        state,
        w,
        strength,
        neutrality_err,
        energy,
        turnover,
        alignment,
        T,
        subspace_angle=0.0,
        factor_leakage=0.0,
    ):
        w_norm = float(np.linalg.norm(w))
        subspace_angle = float(subspace_angle) if subspace_angle is not None else 0.0
        factor_leakage = float(factor_leakage) if factor_leakage is not None else 0.0
        self.Plot("Health", "w_norm", w_norm)
        self.Plot("Health", "energy", energy)
        self.Plot("Health", "turnover", turnover)
        self.Plot("Health", "neutrality_err", neutrality_err)
        self.Plot("Health", "signal_strength", strength)
        self.Plot("Health", "alignment", alignment)
        self.Plot("Health", "subspace_angle", subspace_angle)
        self.Plot("Health", "factor_leakage", factor_leakage)

        self.metric_w_norm.append(w_norm)
        self.metric_energy.append(energy)
        self.metric_turnover.append(turnover)
        self.metric_neutrality.append(neutrality_err)
        self.metric_alignment.append(alignment)
        self.metric_subspace_angle.append(subspace_angle)
        self.metric_factor_leakage.append(factor_leakage)

        self.Debug(
            "{0} | N={1} K={2} T={3} | signal={4:.4f} persist={5} | "
            "w_norm={6:.4f} energy={7:.4f} turnover={8:.4f} | "
            "neutrality_err={9:.6f} | state={10} | angle={11:.5f} | leakage={12:.5f}".format(
                self.Time.date(),
                self.N,
                self.K,
                T,
                strength,
                self.persist_counter,
                w_norm,
                energy,
                turnover,
                neutrality_err,
                state,
                subspace_angle,
                factor_leakage,
            )
        )
