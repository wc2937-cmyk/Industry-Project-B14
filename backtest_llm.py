# backtest_llm.py
import json
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import requests


class BacktestLLMAnalyst:
    """
    对回测结果做自然语言分析的 LLM 助手。
    默认用 deepseek-chat，你可以按需改成自己的 LLM。
    """

    def __init__(
    self,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com/v1/chat/completions",
    model: str = "deepseek-chat",
    verbose: bool = False,
    ):
    self.api_key = api_key
    self.base_url = base_url
    self.model = model
    self.verbose = verbose
    

    # ----------------- 工具函数：把 perf_df + 参数压成文本 -----------------
    def _build_summary_text(
        self,
        perf_df: pd.DataFrame,
        params: Dict,
    ) -> str:
        """
        把绩效表 perf_df 转成适合 LLM 阅读的 summary string。
        perf_df: MultiIndex [Ticker, Model] + [CAGR, Sharpe, MaxDD, AnnRet, AnnVol]
        """
        df = perf_df.copy()

        # 限制一下规模，避免 prompt 太长
        if len(df) > 80:
            df = df.groupby(level=[0, 1]).head(1)

        # 排序一下方便人和 LLM看
        if "Sharpe" in df.columns:
            df_sorted = df.sort_values("Sharpe", ascending=False)
        else:
            df_sorted = df

        table_str = df_sorted.round(4).to_string()

        # 一些简单的统计
        lines = ["Backtest performance table (per ticker, per model):", table_str, ""]

        if "CAGR" in df.columns:
            idx_best_cagr = df["CAGR"].idxmax()
            best_cagr_val = df.loc[idx_best_cagr, "CAGR"]
            lines.append(
                f"Best by CAGR: Ticker={idx_best_cagr[0]}, Model={idx_best_cagr[1]}, "
                f"CAGR={best_cagr_val:.4f}"
            )

        if "Sharpe" in df.columns:
            idx_best_sharpe = df["Sharpe"].idxmax()
            best_sharpe_val = df.loc[idx_best_sharpe, "Sharpe"]
            lines.append(
                f"Best by Sharpe: Ticker={idx_best_sharpe[0]}, Model={idx_best_sharpe[1]}, "
                f"Sharpe={best_sharpe_val:.4f}"
            )

        if "MaxDD" in df.columns:
            idx_worst_dd = df["MaxDD"].idxmin()
            worst_dd_val = df.loc[idx_worst_dd, "MaxDD"]
            lines.append(
                f"Worst drawdown: Ticker={idx_worst_dd[0]}, Model={idx_worst_dd[1]}, "
                f"MaxDD={worst_dd_val:.4f}"
            )

        lines.append("")
        lines.append("Backtest configuration / params:")
        lines.append(json.dumps(params, indent=2))

        return "\n".join(lines)

    # ----------------- 主函数：生成分析 -----------------
    def analyze(
        self,
        perf_df: pd.DataFrame,
        equities: Optional[Dict[Tuple[str, str], pd.Series]] = None,
        params: Optional[Dict] = None,
        language: str = "zh",
    ) -> str:
        """
        perf_df: 你 main() 里构造的绩效表（MultiIndex: [Ticker, Model]）
        equities: {(ticker, model_name) -> equity_curve_series}，可选
        params:   回测参数（hold, fee, state_window 等）
        language: "zh" 输出中文，"en" 输出英文
        """
        if params is None:
            params = {}

        performance_text = self._build_summary_text(perf_df, params)

        # 可选：加一点净值终值信息（不贴整条曲线，避免太长）
        equity_summary = ""
        if equities is not None and len(equities) > 0:
            lines = []
            for (tkr, name), eq in list(equities.items())[:10]:
                if len(eq) == 0:
                    continue
                final_val = float(eq.iloc[-1])
                max_val = float(eq.max())
                min_val = float(eq.min())
                lines.append(
                    f"{tkr} / {name}: final equity={final_val:.3f}, "
                    f"max={max_val:.3f}, min={min_val:.3f}"
                )
            if lines:
                equity_summary = "Equity curve snapshot (final / max / min):\n" + "\n".join(
                    lines
                )

        # ----------------- prompt 设计 -----------------
        if language == "zh":
            system_prompt = (
                "你是一名专业量化投研分析师，擅长解读多因子策略、机器学习模型和强化学习策略的回测结果。"
                "你会拿到一张按 Ticker 和 Model 分组的回测绩效表（包含 CAGR, Sharpe, MaxDD, 年化收益、波动等），"
                "以及部分净值曲线的终值信息和回测参数。"
                "请你输出一份结构清晰的中文分析报告，内容包括：\n"
                "1）整体表现概览：策略组合整体水平，是否有明显 alpha；\n"
                "2）模型对比：哪几类模型（LSTM 回归、传统回归、分类器策略、CQL 策略等）表现最好，侧重点是什么；\n"
                "3）风险收益特征：Sharpe、回撤、波动的平衡情况，是否存在高收益高风险或低波动稳健型模型；\n"
                "4）稳健性与潜在问题：样本长度、过拟合风险、不同 ticker 间表现差异；\n"
                "5）实盘建议：哪些模型适合单独上线，哪些适合做组合；是否建议做模型加权、风控削峰、缩短/拉长持有期等。\n"
                "请用分点+小标题的方式输出，语气偏专业投研，不要太口语化。"
            )
        else:
            system_prompt = (
                "You are a professional quantitative research analyst. "
                "You will receive a performance table for multiple models across multiple tickers, "
                "including CAGR, Sharpe, Max drawdown, annualized return and volatility, "
                "plus some equity-curve summary statistics and backtest configuration. "
                "Write a concise but insightful analysis covering: "
                "(1) overall performance, (2) comparison across model families "
                "(e.g., LSTM regressors, classical regressors, classifiers, CQL RL), "
                "(3) risk/return trade-offs and drawdowns, "
                "(4) robustness and overfitting concerns, "
                "(5) practical deployment suggestions (which models to use, combine, or discard). "
                "Use a professional sell-side/ buy-side research tone."
            )

        user_prompt = (
            "Here is the backtest performance summary:\n\n"
            f"{performance_text}\n\n"
        )
        if equity_summary:
            user_prompt += equity_summary + "\n\n"

        if self.verbose:
            print("======= LLM REQUEST (backtest analysis) =======")
            print("System prompt:\n", system_prompt[:500], "...\n")
            print("User prompt:\n", user_prompt[:500], "...\n")

        # ----------------- 调用 LLM API -----------------
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }

        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return content.strip()

