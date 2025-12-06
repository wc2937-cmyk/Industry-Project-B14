
import numpy as np
import pandas as pd

from Data_handler import MarketDataHandler, GoogleTrendHandler
from Integrate_class import Integrated
from backtest import run_per_ticker, compute_metrics


def main():
    # ========== 1. 配置 ==========
    USE_FAKE_DATA = False  # True 时用假数据快速测试，False 用雅虎+Google Trends

    tickers = ["AAPL", "MSFT", "GOOGL"]
    keywords = ["Apple stock", "Microsoft stock", "Google stock"]

    start_date = "2020-01-01"
    end_date = "2025-01-01"

    # ========== 2. 准备数据：returns / trends / prices_bt ==========
    if USE_FAKE_DATA:
        dates = pd.date_range(start="2020-01-01", periods=200, freq="B")
        rng = np.random.default_rng(0)

        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(len(dates), len(tickers))),
            index=dates,
            columns=tickers,
        )

        trend_df = pd.DataFrame(
            rng.integers(0, 100, size=(len(dates), len(keywords))),
            index=dates,
            columns=keywords,
        )

        prices_bt = (1 + returns).cumprod()

    else:
        # 真实市场数据
        market = MarketDataHandler(tickers, start_date, end_date)
        prices = market.load_data()
        clean_prices = market.clean_data()
        returns = market.compute_returns()      # 同时会保存 returns_data.xlsx

        # Google Trends
        gt = GoogleTrendHandler(keywords, start_date=start_date, end_date=end_date)
        trend_df = gt.load_trends()            # 同时会保存 googletrend_data.xlsx

        # 回测用的价格：clean_prices
        prices_bt = clean_prices

    # ========== 3. 调用 Integrated 训练所有模型（多资产多输出） ==========
    dudu = Integrated()

    model_map = dudu.return_trend(
        returns=returns,
        trends=trend_df,
        tickers=tickers,
        client_text="",
    )


    print("\n===== 训练出的模型类别 =====")
    print("Pred models:", [name for name, _ in model_map["pred"]])
    print("Cls  models:", [name for name, _ in model_map["cls"]])
    print("RL   models:", [name for name, _ in model_map["rl"]])

    # ========== 4. 把模型分配给每只股票 ==========

    all_models = (
        model_map["pred"]
        + model_map["cls"]
        + model_map["rl"]
    )

    models_for_bt = {tkr: all_models for tkr in tickers}

    # ========== 5. 运行回测 ==========
    table, equities, weights = run_per_ticker(
        prices=prices_bt,
        models=models_for_bt,
        strategy="long_short",
        thr_long=0.0,
        thr_short=None,
        hold=10,
        fee_buy=0.0005,
        fee_sell=0.0005,
        leverage_cap=1.0,
        state_window=60,
        warmup_days=0,
        feature_maker=None,     # 使用 backtest.py 里的 default_feature_maker
    )

    # ========== 6. 汇总绩效指标 ==========
    rows = []
    for (tkr, name), eq in equities.items():
        m = compute_metrics(eq)
        rows.append(
            (tkr, name, m["CAGR"], m["Sharpe"], m["MaxDD"], m["AnnRet"], m["AnnVol"])
        )

    perf_df = pd.DataFrame(
        rows,
        columns=["Ticker", "Model", "CAGR", "Sharpe", "MaxDD", "AnnRet", "AnnVol"],
    ).set_index(["Ticker", "Model"])

    print("\n===== Backtest Performance =====\n")
    print(perf_df.round(3))


if __name__ == "__main__":
    main()
