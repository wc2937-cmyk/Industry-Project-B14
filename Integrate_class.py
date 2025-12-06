import yfinance as yf
from pytrends.request import TrendReq
import sys
import pandas as pd
import numpy as np
import os
from CQL_class import *
from LSTM_class import *
from sklearn.preprocessing import StandardScaler
from Self_training_class import *
from Classifier_class import *
import datetime as dt
from LLM import *
from Data_handler import *
class Integrated:
    def __init__(self, llm_assistant: LLMHyperParamAssistant | None = None):
        """
        llm_assistant: 可以传入定义的 LLMHyperParamAssistant 实例，
        如果为 None 就用默认超参数训练。
        """
        self.llm_assistant = llm_assistant

    @staticmethod
    @staticmethod
    def build_cls_labels_from_returns(
            y_raw: np.ndarray,
            scheme: str = "binary",
    ) -> np.ndarray:
        """
        把连续收益率 y_raw 变成离散的分类标签:
        - 支持单资产 (N,) 或多资产 (N, M)
        - 多资产时先做一个简单的 equal-weight 组合，分类目标是组合的方向

        scheme="binary":
            0 = return <= 0
            1 = return > 0

        scheme="ternary":
            0 = 明显跌
            1 = 近平
            2 = 明显涨
        """
        y_raw = np.asarray(y_raw, dtype=float)

        # 多资产：先合成组合收益
        if y_raw.ndim == 2:
            y_raw = y_raw.mean(axis=1)

        y_raw = y_raw.reshape(-1)

        if scheme == "binary":
            labels = (y_raw > 0).astype(int)
        elif scheme == "ternary":
            neg_th = -1e-4
            pos_th = 1e-4
            labels = np.zeros_like(y_raw, dtype=int)
            labels[(y_raw >= neg_th) & (y_raw <= pos_th)] = 1  # flat
            labels[y_raw > pos_th] = 2  # up
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # debug
        print("DEBUG build_cls_labels_from_returns:")
        print("  y_raw sample:", y_raw[:10])
        print("  labels sample:", labels[:10])
        print("  unique labels:", np.unique(labels))

        return labels


    def return_trend(
            self,
            returns: pd.DataFrame,
            trends: pd.DataFrame,
            tickers: list[str],
            client_text: str = "",
    ) -> dict:
        """
        多资产版本：
        - y: 每个 ticker 的下一期收益 (N, M)
        - X: 所有 ticker 的动量/波动率 + 所有 trend 动量 拼成的大特征矩阵 (N, F)
        - 回归 / LSTM / RL 用多输出 y
        - 分类：对 equal-weight 组合做 up/down 标签（长度 N）
        """
        # ===== 1. 多资产特征合并 =====
        merger = DataMerger(returns, trends)
        # 可以在这里调窗口大小，比如 5 日动量 / 波动率
        merger.merge_data(
            ret_mom_window=5,
            ret_vol_window=5,
            trend_mom_window=5,
        )
        merged = merger.merged.copy()

        # ===== 2. 清理 NaN / Inf（rolling 产生的前几行 NaN）=====
        merged = merged.replace([np.inf, -np.inf], np.nan)
        # 只要原始收益或特征有 NaN，就直接丢掉对应行
        cols_for_drop = list(set(merger.return_cols + merger.feature_cols))
        merged = merged.dropna(subset=cols_for_drop)

        # 确保 tickers 和 return_cols 对齐
        ticker_cols = tickers  # 一般就是 merger.return_cols
        for col in ticker_cols:
            if col not in merged.columns:
                raise ValueError(f"Ticker {col} not found in merged columns.")

        # ===== 3. 构造 y（下一期收益）和 X（多资产特征）=====
        # y: 每个 ticker 的下一期收益
        y_all = merged[ticker_cols].shift(-1)

        # 去掉最后一行（shift 之后为 NaN）
        valid_mask = ~y_all.isna().any(axis=1)
        y = y_all[valid_mask]
        # 特征：去掉原始收益列，保留所有 engineered features + 原始 trend
        X = merged.loc[valid_mask.index[valid_mask]].drop(columns=ticker_cols)

        # 再防守性清一遍 NaN/Inf（理论上不应该有了）
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        # y 只保留 X 还在的那些 index（防止 dropna 再不对齐）
        y = y.loc[X.index]

        print("=== FINAL DATA SHAPE ===")
        print(f"X shape: {X.shape}")  # (N, F)
        print(f"y shape: {y.shape}")  # (N, M)

        # ===== 4. 标准化特征 X =====
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ===== 5. 从 LLM 或默认值加载全量 config =====
        if self.llm_assistant is not None:
            # 可选：传一点当前数据的摘要进去
            data_summary = (
                f"X shape: {X_scaled.shape}, y shape: {y.values.shape}, "
                f"tickers: {tickers}, feature_cols: {list(X.columns)[:10]}..."
            )
            cfg = self.llm_assistant.suggest_config(
                client_text,
                code_files=[
                    "Integrate_class.py",
                    "LSTM_class.py",
                    "CQL_class.py",
                    "Classifier_class.py",
                    "Self_training_class.py",
                ],
                data_summary=data_summary,
            )
        else:
            cfg = {
                "lstm": {
                    "enable": True,
                    "lr": 1e-3,
                    "max_epochs": 30,
                    "hidden_dim1": 64,
                    "hidden_dim2": 32,
                    "batch_size": 64,
                    "train_ratio": 0.8,
                },
                "pred": {
                    "enable": True,
                    "train_ratio": 0.8,
                    "lr_params": {},
                    "rf_params": {"n_estimators": 300},
                    "br_params": {},
                },
                "cls": {
                    "enable": True,
                    "train_ratio": 0.8,
                    "xgb_params": {"max_depth": 6},
                    "rf_params": {"n_estimators": 300},
                },
                "rl": {
                    "enable": True,
                    "train_ratio": 0.8,
                    "n_steps": 3000,
                    "n_steps_per_epoch": 1000,
                    "batch_size": 256,
                    "cql_config_kwargs": {},
                },
            }

        lstm_cfg = cfg.get("lstm", {})
        pred_cfg = cfg.get("pred", {})
        cls_cfg = cfg.get("cls", {})
        rl_cfg = cfg.get("rl", {})

        # ===== 6. LSTM 多输出回归 =====
        pred_models = []

        if lstm_cfg.get("enable", True):
            trainer_lstm = LSTMRegressorTrainer(
                lr=lstm_cfg.get("lr", 1e-3),
                max_epochs=int(lstm_cfg.get("max_epochs", 30)),
                hidden_dim1=int(lstm_cfg.get("hidden_dim1", 64)),
                hidden_dim2=int(lstm_cfg.get("hidden_dim2", 32)),
                batch_size=int(lstm_cfg.get("batch_size", 64)),
            )
            models_lstm = trainer_lstm.fit(
                X_scaled,
                y.values,
                train_ratio=float(lstm_cfg.get("train_ratio", 0.8)),
                lr=lstm_cfg.get("lr", None),
                max_epochs=lstm_cfg.get("max_epochs", None),
                hidden_dim1=lstm_cfg.get("hidden_dim1", None),
                hidden_dim2=lstm_cfg.get("hidden_dim2", None),
                batch_size=lstm_cfg.get("batch_size", None),
            )
            pred_models.extend(models_lstm)

        # ===== 7. CQL policy（基于组合收益的 RL）=====
        models_rl = []
        if rl_cfg.get("enable", True):
            trainer_cql = CQLPolicyTrainer(
                train_ratio=float(rl_cfg.get("train_ratio", 0.8)),
                n_steps=int(rl_cfg.get("n_steps", 3000)),
                n_steps_per_epoch=int(rl_cfg.get("n_steps_per_epoch", 1000)),
                batch_size=int(rl_cfg.get("batch_size", 256)),
                cql_config_kwargs=rl_cfg.get("cql_config_kwargs", {}),
            )

            models_rl = trainer_cql.fit(
                X_scaled,
                y.values,
                train_ratio=float(rl_cfg.get("train_ratio", 0.8)),
                n_steps=rl_cfg.get("n_steps", None),
                n_steps_per_epoch=rl_cfg.get("n_steps_per_epoch", None),
                batch_size=rl_cfg.get("batch_size", None),
                cql_config_kwargs=rl_cfg.get("cql_config_kwargs", None),
            )

        # ===== 8. 传统回归器 TriRegressorTrainer =====
        if pred_cfg.get("enable", True):
            trainer_pred = TriRegressorTrainer(
                default_lr_params=pred_cfg.get("lr_params", {}),
                default_rf_params=pred_cfg.get("rf_params", {}),
                default_br_params=pred_cfg.get("br_params", {}),
            )

            models_pred_reg = trainer_pred.fit(
                X_scaled,
                y.values,
                train_ratio=float(pred_cfg.get("train_ratio", 0.8)),
                lr_params=pred_cfg.get("lr_params", None),
                rf_params=pred_cfg.get("rf_params", None),
                br_params=pred_cfg.get("br_params", None),
            )
            pred_models.extend(models_pred_reg)

        # ===== 9. 分类器：对组合收益做 up/down 标签 =====
        models_cls = []
        if cls_cfg.get("enable", True):
            trainer_cls = TriClassifierTrainer(
                default_xgb_params=cls_cfg.get("xgb_params", {}),
                default_rf_params=cls_cfg.get("rf_params", {}),
            )

            y_cls_raw = y.values  # (N, M) 多资产收益
            y_cls = self.build_cls_labels_from_returns(y_cls_raw, scheme="binary")

            models_cls = trainer_cls.fit(
                X_scaled,
                y_cls,
                train_ratio=float(cls_cfg.get("train_ratio", 0.8)),
                xgb_params=cls_cfg.get("xgb_params", None),
                rf_params=cls_cfg.get("rf_params", None),
            )

        # ===== 10. 合并返回 =====
        return {
            "pred": pred_models,
            "cls": models_cls,
            "rl": models_rl,
        }



if __name__ == "__main__":

    # ========== 1. 配置：使用假数据还是真实数据 ==========
    USE_FAKE_DATA = False

    tickers = ["AAPL", "MSFT", "GOOGL"]
    keywords = ["Apple stock", "Microsoft stock", "Google stock"]

    # ========== 2. 构造 returns 和 trend_df ==========
    if USE_FAKE_DATA:
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        rng = np.random.default_rng(0)

        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(200, len(tickers))),
            index=dates,
            columns=tickers,
        )

        trend_data = rng.integers(0, 100, size=(200, len(keywords)))
        trend_df = pd.DataFrame(trend_data, index=dates, columns=keywords)

        print("Fake data generation complete.")

    else:
        market = MarketDataHandler(tickers, "2020-01-01", "2025-01-01")
        prices = market.load_data()
        clean_prices = market.clean_data()
        returns = market.compute_returns()

        gt = GoogleTrendHandler(keywords)
        trend_df = gt.load_trends()

    # ========== 3. 初始化 LLM 超参助手 ==========
    llm_helper = LLMHyperParamAssistant(
        api_key="sk-161c540117774127bd9a7f87fad87727",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        verbose=True,
    )

    # ========== 4. 构造 Integrated ==========
    dudu = Integrated(llm_assistant=llm_helper)

    # ========== 5. 上一层 client 的文字描述 ==========
    client_words = (
        "Please focus on stable training. Avoid overfitting. "
        "Data is noisy. Prefer smoother hyperparameters."
    )

    # ========== 6. 训练 & 生成模型 map ==========
    model_map = dudu.return_trend(
        returns=returns,
        trends=trend_df,
        tickers=tickers,
        client_text=client_words,
    )

    # ========== 7. 输出三个类别模型名称 ==========
    print("\n================ Result ================")
    print("Pred Models:")
    print([name for name, _ in model_map["pred"]])

    print("\nClassification Models:")
    print([name for name, _ in model_map["cls"]])

    print("\nRL Models:")
    print([name for name, _ in model_map["rl"]])
