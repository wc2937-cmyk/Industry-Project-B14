import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
class MarketDataHandler:
    def __init__(self, tickers, start, end, save_path="market_data.xlsx"):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.save_path = save_path
        self.data = None
        self.returns = None

    def load_data(self):
        raw = yf.download(self.tickers, start=self.start, end=self.end, auto_adjust=True)
        self.data = raw["Close"].copy()
        print(f"Loaded {len(self.data)} days of data for {len(self.tickers)} tickers.")
        return self.data

    def clean_data(self):
        self.data = self.data.dropna().astype(float)
        print(f"Cleaned data — shape: {self.data.shape}")
        return self.data

    def compute_returns(self):
        self.returns = self.data.pct_change().dropna()
        self.returns.to_excel("returns_data.xlsx")
        print("Saved returns_data.xlsx")
        return self.returns

    def save_to_excel(self):
        self.data.to_excel(self.save_path)
        print(f"Market data saved to {self.save_path}")
class DataMerger:
    def __init__(self, returns_df, trends_df, keep="left"):
        """
        returns_df: 日频(交易日)收益率，多列为各 ticker
        trends_df:  周/日频 Google Trends，多列为各关键词
        keep: 'left' 用股票交易日为主；也可选 'inner'
        """
        self.returns = returns_df.copy()
        self.trends = trends_df.copy()
        self.keep = keep
        self.merged = None

        # 方便上层使用
        self.return_cols = list(self.returns.columns)
        self.trend_cols = list(self.trends.columns)
        self.feature_cols = []

    def _normalize_dtindex(self, df):
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = pd.to_datetime(idx.date)
        out = df.copy()
        out.index = idx
        return out

    def merge_data(
        self,
        ret_mom_window: int = 5,
        ret_vol_window: int = 5,
        trend_mom_window: int = 5,
    ):
        """
        多资产版本的合并 + 特征工程：
        - 对每个 ticker 独立算动量 / 波动率：
            <TICKER>_ret_mom_5d, <TICKER>_ret_vol_5d
        - 对每个 trend 列算动量：
            <TREND_COL>_trend_mom_5d
        合并结果放在 self.merged，原始收益列保持不变。
        """
        # ===== 1. 对齐日期 =====
        r = self._normalize_dtindex(self.returns)
        t = self._normalize_dtindex(self.trends)

        # 把 Google Trends 变成工作日频，向前填充
        t = t.resample("B").ffill()

        start = max(r.index.min(), t.index.min())
        end = min(r.index.max(), t.index.max())
        r = r.loc[start:end]
        t = t.loc[start:end]

        if self.keep == "inner":
            merged = r.join(t, how="inner")
        else:
            merged = r.join(t, how="left")

        # 再次向前/向后填充 trend 缺失值
        trend_cols = [c for c in merged.columns if c not in r.columns]
        merged[trend_cols] = merged[trend_cols].ffill().bfill()

        # ===== 2. 为每个 ticker 单独算特征 =====
        feature_cols = []

        # 每个资产：动量 & 波动率
        for col in r.columns:
            mom_col = f"{col}_ret_mom_{ret_mom_window}d"
            vol_col = f"{col}_ret_vol_{ret_vol_window}d"

            merged[mom_col] = (
                merged[col]
                .rolling(ret_mom_window, min_periods=ret_mom_window)
                .mean()
            )
            merged[vol_col] = (
                merged[col]
                .rolling(ret_vol_window, min_periods=ret_vol_window)
                .std()
            )

            feature_cols += [mom_col, vol_col]

        # 每个趋势列：趋势动量
        for col in trend_cols:
            mom_col = f"{col}_trend_mom_{trend_mom_window}d"
            merged[mom_col] = (
                merged[col]
                .pct_change()
                .rolling(trend_mom_window, min_periods=trend_mom_window)
                .mean()
            )
            feature_cols.append(mom_col)

        # 不在这里 dropna，让上层统一处理
        print("=== MERGE DIAGNOSTICS (multi-asset) ===")
        print(f"returns:  {r.index.min()} → {r.index.max()}  rows={len(r)}  cols={len(r.columns)}")
        print(f"trends:   {t.index.min()} → {t.index.max()}  rows={len(t)}  cols={len(t.columns)}")
        print(f"overlap:  {merged.index.min()} → {merged.index.max()}  rows={len(merged)}")
        print(f"return_cols: {list(r.columns)}")
        print(f"trend_cols:  {trend_cols}")
        print(f"feature_cols: {feature_cols}")

        merged.to_excel("merged_features_multiasset.xlsx")
        print("Merged dataset saved to merged_features_multiasset.xlsx")

        self.merged = merged
        self.return_cols = list(r.columns)
        self.trend_cols = trend_cols
        self.feature_cols = feature_cols

# =============================
class GoogleTrendHandler:
    def __init__(self, keywords, start_date="2020-01-01", end_date="2025-01-01", geo="US"):
        self.keywords = keywords
        self.start_date = start_date
        self.end_date = end_date
        self.geo = geo
        self.trends = None

    def load_trends(self):
        pytrends = TrendReq()
        pytrends.build_payload(self.keywords, timeframe=f"{self.start_date} {self.end_date}", geo=self.geo)
        self.trends = pytrends.interest_over_time()
        if 'isPartial' in self.trends.columns:
            self.trends = self.trends.drop(columns=['isPartial'])
        self.trends = self.trends.fillna(method='ffill')
        self.trends.to_excel("googletrend_data.xlsx")
        print("Google Trends data saved to googletrend_data.xlsx")
        return self.trends