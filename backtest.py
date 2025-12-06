from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def default_feature_maker(price_df: pd.DataFrame,
                          state_window: int = 60,
                          tickers_for_drop: Optional[List[str]] = None):
    """
    Multi-asset 默认特征生成器：
    - 读取 Integrated_class.py 生成的 merged_features_multiasset.xlsx
    - 用其中的 ticker 列构造 y
    - 把这些 ticker 列从特征里删掉，得到 X
    - 再做一次标准化，保证和原 backtest 逻辑一致
    """

    import os
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    xlsx_path = "merged_features_multiasset.xlsx"
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            "merged_features_multiasset.xlsx not found. "
            "Please run Integrated.return_trend() first."
        )

    merged = pd.read_excel(xlsx_path, index_col=0, parse_dates=True)

    # 检查 ticker 列
    if tickers_for_drop is None:
        ticker_cols = [c for c in merged.columns if c in price_df.columns]
    else:
        ticker_cols = [c for c in merged.columns if c in tickers_for_drop]

    if not ticker_cols:
        raise ValueError(
            f"These ticker columns were not found in merged_features_multiasset.xlsx: "
            f"{tickers_for_drop or list(price_df.columns)}"
        )

    y = merged[ticker_cols].shift(-1).dropna()

    X = merged.loc[y.index].drop(columns=ticker_cols)

    # 清理 inf
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns
    )

    # index 对齐 price_df，删掉缺失行
    X_scaled = X_scaled.reindex(price_df.index).dropna()

    return X_scaled, X_scaled.index




def compute_metrics(equity: pd.Series, risk_free: float = 0.0) -> Dict[str, float]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "AnnRet": np.nan, "AnnVol": np.nan}
    rets = eq.pct_change().dropna()
    ann_fac = 252.0
    cum_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (1.0 + cum_ret) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    ann_ret = float(rets.mean() * ann_fac)
    ann_vol = float(rets.std(ddof=0) * np.sqrt(ann_fac))
    sharpe = (ann_ret - risk_free) / (ann_vol + 1e-12) if ann_vol > 0 else np.nan
    roll_max = eq.cummax()
    dd = (eq / (roll_max + 1e-12)) - 1.0
    maxdd = float(dd.min())
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd, "AnnRet": ann_ret, "AnnVol": ann_vol}

# Model Adapter: Unified Interface

class ModelAdapter:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        if hasattr(model, "predict_proba"):
            self.kind = "classifier"
        elif hasattr(model, "act"):
            self.kind = "rl"
        else:
            self.kind = "predictor"

    def infer(self, X, col_idx: int | None = None, col_name: str | None = None):
        if self.kind == "classifier":
            return self.infer_classifier(X)
        elif self.kind == "rl":
            return self.infer_rl(X)
        else:
            return self.infer_predictor(X, col_idx=col_idx, col_name=col_name)

    def infer_predictor(self, X, col_idx: int | None = None, col_name: str | None = None):
        y = self.model.predict(np.asarray(X))
        if isinstance(y, pd.Series):
            y1 = y.to_numpy()
        elif isinstance(y, pd.DataFrame):
            if (col_name is not None) and (col_name in y.columns):
                y1 = y[col_name].to_numpy()
            elif (col_idx is not None) and (0 <= col_idx < y.shape[1]):
                y1 = y.iloc[:, col_idx].to_numpy()
            else:
                y1 = y.iloc[:, 0].to_numpy()
        else:
            y = np.asarray(y)
            if y.ndim == 2:
                if y.shape[1] == 1:
                    y1 = y[:, 0]
                else:
                    i = 0 if col_idx is None else int(col_idx)
                    if i >= y.shape[1]:
                        i = y.shape[1] - 1
                    y1 = y[:, i]
            else:
                y1 = y

        return pd.Series(y1, index=X.index, name=f"{self.name}_yhat")


    def infer_classifier(self, X):
        proba = self.model.predict_proba(X)
        if isinstance(proba, pd.DataFrame):
            p1 = proba[1].to_numpy() if 1 in proba.columns else proba.iloc[:, -1].to_numpy()
        else:
            proba = np.asarray(proba)
            p1 = proba[:, -1] if proba.ndim == 2 else proba
        return pd.Series(p1, index=X.index, name=f"{self.name}_p_up")


    def infer_rl(self, obs: pd.DataFrame) -> pd.Series:
        if hasattr(self.model, "act_batch"):
            a = self.model.act_batch(obs)
        elif hasattr(self.model, "predict"):
            a = self.model.predict(obs)
        elif hasattr(self.model, "act"):
            a = np.array([self.model.act(obs.iloc[i : i + 1]) for i in range(len(obs))])
        else:
            raise ValueError(f"RL model {self.name} has no valid interface.")
        a = np.clip(np.round(np.asarray(a).reshape(-1)), -1, 1)
        return pd.Series(a, index=obs.index, name=f"{self.name}_action")


# Signal generation

def signal_from_predictor(yhat: pd.Series, thr_long: float = 0.0, thr_short: Optional[float] = None) -> pd.Series:
    thr_short = -thr_long if thr_short is None else thr_short
    sig = pd.Series(0.0, index=yhat.index)
    sig[yhat >= thr_long] = 1.0
    sig[yhat <= thr_short] = -1.0
    return sig


def signal_from_classifier(p_up: pd.Series, tau: float = 0.5, band: float = 0.0, allow_short: bool = False) -> pd.Series:
    long_thr = max(tau, 0.5 + band)
    short_thr = min(1.0 - tau, 0.5 - band)
    sig = pd.Series(0.0, index=p_up.index)
    sig[p_up >= long_thr] = 1.0
    if allow_short:
        sig[p_up <= short_thr] = -1.0
    return sig

# Backtesting engine

def backtest_hold_H(prices: pd.Series, signal: pd.Series, H: int = 10,
                    fee_buy: float = 0.0, fee_sell: float = 0.0,
                    clip_leverage: float = 1.0):
    px = prices.astype(float)
    rets = px.pct_change().fillna(0.0)
    sig = signal.reindex(px.index).fillna(0.0)

    w = pd.Series(0.0, index=px.index)
    idx = px.index
    for t in range(len(idx) - 1):
        s = float(sig.iloc[t])
        if s == 0.0: continue
        start, end = t + 1, min(t + H, len(idx) - 1)
        w.iloc[start:end + 1] += s
    w = w.clip(-clip_leverage, clip_leverage)

    equity = pd.Series(index=px.index, dtype=float)
    equity.iloc[0] = 1.0
    prev_w = w.iloc[0]
    trade_log = []
    for i in range(1, len(px)):
        dw = float(w.iloc[i] - prev_w)
        buy, sell = max(dw, 0.0), max(-dw, 0.0)
        cost = buy * fee_buy + sell * fee_sell
        growth = prev_w * rets.iloc[i]
        equity.iloc[i] = equity.iloc[i - 1] * (1.0 + growth) * (1.0 - cost)
        if buy + sell > 0:
            trade_log.append({"date": idx[i], "buy": buy, "sell": sell, "cost": cost})
        prev_w = w.iloc[i]
    return w, equity, pd.DataFrame(trade_log).set_index("date") if trade_log else pd.DataFrame()


def backtest_daily_actions(prices: pd.Series, actions: pd.Series,
                           fee_buy: float = 0.0, fee_sell: float = 0.0,
                           clip_leverage: float = 1.0):
    px = prices.astype(float)
    rets = px.pct_change().fillna(0.0)
    act = actions.reindex(px.index).fillna(0.0).clip(-clip_leverage, clip_leverage)
    w = act.shift(1).fillna(0.0)

    equity = pd.Series(index=px.index, dtype=float)
    equity.iloc[0] = 1.0
    prev_w = w.iloc[0]
    trade_log = []
    for i in range(1, len(px)):
        dw = float(w.iloc[i] - prev_w)
        buy, sell = max(dw, 0.0), max(-dw, 0.0)
        cost = buy * fee_buy + sell * fee_sell
        growth = prev_w * rets.iloc[i]
        equity.iloc[i] = equity.iloc[i - 1] * (1.0 + growth) * (1.0 - cost)
        if buy + sell > 0:
            trade_log.append({"date": px.index[i], "buy": buy, "sell": sell, "cost": cost})
        prev_w = w.iloc[i]
    return w, equity, pd.DataFrame(trade_log).set_index("date") if trade_log else pd.DataFrame()

# Main function: Single ticket backtesting
def run_per_ticker(
    prices: pd.DataFrame,
    models,
    strategy: str = "long_short",
    thr_long: float = 0.0,
    thr_short: Optional[float] = None,
    hold: int = 1,
    H: Optional[int] = None,
    fee_buy: float = 0.0,
    fee_sell: float = 0.0,
    leverage_cap: float = 1.0,
    state_window: int = 60,
    warmup_days: int = 0,
    feature_maker: Optional[
        Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.Index]]
    ] = None,
    model_adapter_class=None,
    **kwargs,
):
    assert isinstance(prices, pd.DataFrame) and prices.shape[1] > 0
    tickers_all = list(prices.columns)

    if H is not None:
        hold = int(H)

    # Unified Model Input Format
    def _norm_pairs_list(obj):
        pairs = []
        if isinstance(obj, (list, tuple)):
            for i, m in enumerate(obj):
                if isinstance(m, (list, tuple)) and len(m) == 2:
                    pairs.append((str(m[0]), m[1]))
                else:
                    pairs.append((f"model_{i}", m))
        else:
            pairs = [(getattr(obj, "__name__", "model"), obj)]
        return pairs

    if isinstance(models, dict):
        models_map: Dict[str, List[Tuple[str, Any]]] = {}
        for tkr, v in models.items():
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple)):
                models_map[tkr] = [(str(n), m) for (n, m) in v]
            elif isinstance(v, (list, tuple)) and len(v) == 2 and not isinstance(v[0], (list, tuple)):
                models_map[tkr] = [(str(v[0]), v[1])]
            else:
                models_map[tkr] = _norm_pairs_list(v)
        target_tickers = list(models_map.keys())
    elif isinstance(models, (list, tuple)):
        pairs_all = _norm_pairs_list(models)
        target_tickers = tickers_all
        models_map = {t: pairs_all for t in target_tickers}
    else:
        raise TypeError("`models` must be either a Dict[...] or a List[...]")

    Adapter = model_adapter_class or ModelAdapter

    ret_cols = []
    ret_series_list = []
    equities: Dict[Tuple[str, str], pd.Series] = {}
    weights: Dict[Tuple[str, str], pd.Series] = {}

    for tkr in target_tickers:
        px = prices[[tkr]].dropna()

        if feature_maker is not None:
            X, dates = feature_maker(px)
        else:
            X, dates = default_feature_maker(prices, state_window, tickers_for_drop=tickers_all)

        for name_i, model_i in models_map[tkr]:
            adp = Adapter(name_i, model_i)
            col_idx = tickers_all.index(tkr) if tkr in tickers_all else None
            yser = adp.infer(X, col_idx=col_idx, col_name=tkr)
            kind = getattr(adp, "kind", "predictor")

            # Classifier Model
            if kind == "classifier":
                tau = float(kwargs.get("tau", 0.5))
                band = float(kwargs.get("band", 0.0))
                allow_short = bool(kwargs.get("allow_short", False))
                pos = signal_from_classifier(yser, tau=tau, band=band, allow_short=allow_short)
                w, eq, _ = backtest_hold_H(
                    prices=px[tkr], signal=pos, H=hold,
                    fee_buy=fee_buy, fee_sell=fee_sell, clip_leverage=leverage_cap
                )

            # Reinforcement Learning Model
            elif kind == "rl":
                actions = yser.clip(-leverage_cap, leverage_cap)
                w, eq, _ = backtest_daily_actions(
                    prices=px[tkr], actions=actions,
                    fee_buy=fee_buy, fee_sell=fee_sell, clip_leverage=leverage_cap
                )

            # Predictor Model
            else:
                if strategy == "proportional":
                    pos = yser.clip(-1, 1)
                elif strategy == "long_only":
                    pos = (yser > thr_long).astype(float)
                else:  # long_short
                    thr_s = (-thr_long) if (thr_short is None) else float(thr_short)
                    pos = pd.Series(0.0, index=yser.index)
                    pos[yser > thr_long] = 1.0
                    pos[yser < thr_s] = -1.0

                w, eq, _ = backtest_hold_H(
                    prices=px[tkr], signal=pos, H=hold,
                    fee_buy=fee_buy, fee_sell=fee_sell, clip_leverage=leverage_cap
                )

            strat_ret = eq.pct_change().fillna(0.0).rename((tkr, name_i))
            ret_series_list.append(strat_ret)
            ret_cols.append((tkr, name_i))
            equities[(tkr, name_i)] = eq.rename((tkr, name_i))
            weights[(tkr, name_i)] = w.rename((tkr, name_i))

    # Summary Output
    if ret_series_list:
        table = pd.concat(ret_series_list, axis=1)
        table.columns = pd.MultiIndex.from_tuples(ret_cols, names=["ticker", "model"])
        table = table.fillna(0.0)
    else:
        table = pd.DataFrame(index=prices.index.copy())

    return table, equities, weights




