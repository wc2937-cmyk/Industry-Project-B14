from __future__ import annotations
from typing import List, Tuple, Union, Optional, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge

ModelItem = Tuple[str, object]

class TriRegressorTrainer:
    def __init__(
        self,
        default_lr_params: Optional[Dict[str, Any]] = None,
        default_rf_params: Optional[Dict[str, Any]] = None,
        default_br_params: Optional[Dict[str, Any]] = None,
    ):
        """
        default_*_params: 可选的“全局默认超参数”，fit 时还能再覆盖。
        例如：
        TriRegressorTrainer(default_rf_params={"max_depth": 5})
        """
        self.models_: List[ModelItem] = []
        self.train_ratio_: float = 0.8
        self.default_lr_params = default_lr_params or {}
        self.default_rf_params = default_rf_params or {}
        self.default_br_params = default_br_params or {}

    @staticmethod
    def _to_2d(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _to_y(y):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            return y
        return y

    def fit(
            self,
            X,
            y,
            train_ratio: float = 0.8,
            random_state: int = 42,
            shuffle: bool = False,
            lr_params: Optional[Dict[str, Any]] = None,
            rf_params: Optional[Dict[str, Any]] = None,
            br_params: Optional[Dict[str, Any]] = None,
    ) -> List[ModelItem]:
        """
        支持 y 一维或二维:
        - y.shape = (N,)      → 单资产
        - y.shape = (N, M)    → 多资产 (multi-output)
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)  # 统一成 (N, M)

        if not (0 < train_ratio <= 1):
            raise ValueError("train_ratio 必须在 (0,1] 之间。")

        N = len(X)
        split = int(N * train_ratio)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]  # y_tr: (N_tr, M)

        # ===== 组装原始超参 =====
        lr_cfg_raw = dict(self.default_lr_params)
        if lr_params:
            lr_cfg_raw.update(lr_params)

        rf_cfg_raw = {
            "n_estimators": 300,
            "random_state": random_state,
            "n_jobs": -1,
        }
        rf_cfg_raw.update(self.default_rf_params)
        if rf_params:
            rf_cfg_raw.update(rf_params)

        br_cfg_raw = dict(self.default_br_params)
        if br_params:
            br_cfg_raw.update(br_params)

        # ===== 白名单过滤 =====
        valid_lr = {"fit_intercept", "copy_X", "n_jobs", "positive"}
        valid_rf = {
            "n_estimators", "criterion", "max_depth",
            "min_samples_split", "min_samples_leaf",
            "min_weight_fraction_leaf", "max_features",
            "max_leaf_nodes", "min_impurity_decrease",
            "bootstrap", "oob_score", "n_jobs", "random_state",
        }
        valid_br = {
            "tol", "alpha_1", "alpha_2", "lambda_1", "lambda_2",
            "alpha_init", "lambda_init", "compute_score",
            "fit_intercept", "copy_X", "verbose",
        }

        def filt(d, valid):
            return {k: v for k, v in d.items() if k in valid}

        lr_cfg = filt(lr_cfg_raw, valid_lr)
        rf_cfg = filt(rf_cfg_raw, valid_rf)
        br_cfg = filt(br_cfg_raw, valid_br)

        unused = {
            "lr": {k: v for k, v in lr_cfg_raw.items() if k not in valid_lr},
            "rf": {k: v for k, v in rf_cfg_raw.items() if k not in valid_rf},
            "br": {k: v for k, v in br_cfg_raw.items() if k not in valid_br},
        }
        for tag, d in unused.items():
            if d:
                print(f"[Regressor WARNING] Invalid {tag} keys:", d)

        # ===== 1) LinearRegression / RF：直接支持 multi-output =====
        lr = LinearRegression(**lr_cfg).fit(X_tr, y_tr)
        rf = RandomForestRegressor(**rf_cfg).fit(X_tr, y_tr)

        # ===== 2) BayesianRidge：每一列一个模型，然后包一层 wrapper =====
        M = y_tr.shape[1]
        br_models = []
        for j in range(M):
            br_j = BayesianRidge(**br_cfg)
            br_j.fit(X_tr, y_tr[:, j])
            br_models.append(br_j)

        class _BRMultiWrapper:
            def __init__(self, models):
                self.models = models

            def predict(self, X):
                X = np.asarray(X, float)
                preds = [m.predict(X).reshape(-1, 1) for m in self.models]
                return np.hstack(preds)  # (N, M)

        br = _BRMultiWrapper(br_models)

        self.models_ = [
            ("LinearRegression", lr),
            ("RandomForestRegressor", rf),
            ("BayesianRidge", br),
        ]
        return self.models_


