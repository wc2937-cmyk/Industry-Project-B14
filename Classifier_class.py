from __future__ import annotations
from typing import List, Tuple, Union, Optional, Dict, Any
from pathlib import Path

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ModelItem = Tuple[str, object]


class TriClassifierTrainer:
    def __init__(
        self,
        default_xgb_params: Optional[Dict[str, Any]] = None,
        default_rf_params: Optional[Dict[str, Any]] = None,
    ):
        """
        default_xgb_params / default_rf_params:
          全局默认超参，fit 时还能传入 xgb_params, rf_params 再覆盖。
        """
        self.models_: List[ModelItem] = []
        self.train_ratio_: float = 0.8
        self.default_xgb_params = default_xgb_params or {}
        self.default_rf_params = default_rf_params or {}

    @staticmethod
    def _to_2d(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    @staticmethod
    def _to_y(y):
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()
        return y

    def fit(
        self,
        X,
        y,
        train_ratio: float = 0.8,
        random_state: int = 42,
        shuffle: bool = False,
        xgb_params: Optional[Dict[str, Any]] = None,
        rf_params: Optional[Dict[str, Any]] = None,
    ) -> List[ModelItem]:
        """
        xgb_params: 覆盖 XGBClassifier 参数
        rf_params:  覆盖 RandomForestClassifier 参数
        """
        if not (0 < train_ratio <= 1):
            raise ValueError("train_ratio 必须在 (0, 1] 之间。")

        Xn = self._to_2d(X)
        yn = self._to_y(y)
        self.train_ratio_ = float(train_ratio)

        if train_ratio < 1.0:
            X_tr, X_val, y_tr, y_val = train_test_split(
                Xn,
                yn,
                test_size=(1.0 - train_ratio),
                random_state=random_state,
                shuffle=shuffle,
            )
        else:
            X_tr, y_tr = Xn, yn

        # ===== XGB 原始配置 =====
        xgb_cfg_raw = {
            "random_state": random_state,
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
        }
        xgb_cfg_raw.update(self.default_xgb_params)
        if xgb_params:
            xgb_cfg_raw.update(xgb_params)

        # ===== RF 原始配置 =====
        rf_cfg_raw = {
            "n_estimators": 300,
            "random_state": random_state,
            "n_jobs": -1,
        }
        rf_cfg_raw.update(self.default_rf_params)
        if rf_params:
            rf_cfg_raw.update(rf_params)

        # ===== 白名单过滤，避免 LLM 乱塞参数 =====
        valid_xgb = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "gamma",
            "min_child_weight",
            "reg_lambda",
            "reg_alpha",
            "scale_pos_weight",
            "max_delta_step",
            "random_state",
            "eval_metric",
            "n_jobs",
        }
        valid_rf = {
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "bootstrap",
            "oob_score",
            "n_jobs",
            "random_state",
            "class_weight",
        }

        def filt(d, valid):
            return {k: v for k, v in d.items() if k in valid}

        xgb_cfg = filt(xgb_cfg_raw, valid_xgb)
        rf_cfg = filt(rf_cfg_raw, valid_rf)

        unused_xgb = {k: v for k, v in xgb_cfg_raw.items() if k not in valid_xgb}
        unused_rf = {k: v for k, v in rf_cfg_raw.items() if k not in valid_rf}

        if unused_xgb:
            print("[Classifier WARNING] Invalid XGBClassifier keys:", unused_xgb)
        if unused_rf:
            print("[Classifier WARNING] Invalid RFClassifier keys:", unused_rf)

        xgb_clf = XGBClassifier(**xgb_cfg)
        xgb_clf.fit(X_tr, y_tr)

        rf_clf = RandomForestClassifier(**rf_cfg)
        rf_clf.fit(X_tr, y_tr)

        self.models_ = [
            ("XGBClassifier", xgb_clf),
            ("RandomForestClassifier", rf_clf),
        ]
        return list(self.models_)

    def save(self, path: Optional[Union[str, Path]] = None) -> List[Path]:
        if not self.models_:
            raise RuntimeError("没有已训练模型：请先调用 fit()。")
        out_dir = Path(path) if path else Path("./ML")
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []
        for name, model in self.models_:
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            fpath = out_dir / f"{safe}.joblib"
            joblib.dump(model, fpath)
            saved_paths.append(fpath)
        return saved_paths
