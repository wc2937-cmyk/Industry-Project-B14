
import numpy as np
import pandas as pd

from Data_handler import MarketDataHandler, GoogleTrendHandler
from Integrate_class import Integrated
from backtest import run_per_ticker, compute_metrics


def main():
    # ========== 1. Configuration ==========
    USE_FAKE_DATA = False  # When True, use fake data for quick testing; when False, use Yahoo! + Google Trends.

    tickers = ["AAPL", "MSFT", "GOOGL"]
    keywords = ["Apple stock", "Microsoft stock", "Google stock"]

    start_date = "2020-01-01"
    end_date = "2025-01-01"

    # ========== 2. Data Preparation: returns / trends / prices_bt ==========
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
        # Real Market Data
        market = MarketDataHandler(tickers, start_date, end_date)
        prices = market.load_data()
        clean_prices = market.clean_data()
        returns = market.compute_returns()     # This will also save returns_data.xlsx

        # Google Trends
        gt = GoogleTrendHandler(keywords, start_date=start_date, end_date=end_date)
        trend_df = gt.load_trends()           # It will also save googletrend_data.xlsx

        #Backtesting price: clean_prices
        prices_bt = clean_prices

    # ========== 3. Call Integrated to train all models (multiple assets, multiple outputs) ==========
    dudu = Integrated()

    model_map = dudu.return_trend(
        returns=returns,
        trends=trend_df,
        tickers=tickers,
        client_text="",
    )


    print("\n===== Training model categories=====")
    print("Pred models:", [name for name, _ in model_map["pred"]])
    print("Cls  models:", [name for name, _ in model_map["cls"]])
    print("RL   models:", [name for name, _ in model_map["rl"]])

    # ========== 4. Assign the model to each stock==========

    all_models = (
        model_map["pred"]
        + model_map["cls"]
        + model_map["rl"]
    )

    models_for_bt = {tkr: all_models for tkr in tickers}

    # ========== 5. Run backtest ==========
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
        feature_maker=None,     # Use the default_feature_maker in backtest.py
    )

    # ========== 6. Summary of performance indicators==========
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

# ========== 7. Use LLM to comment on the backtest results.==========
    try:
        from backtest_llm import BacktestLLMAnalyst

        analyst = BacktestLLMAnalyst(
            api_key="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",
            base_url="https://api.groq.com/openai/v1/chat/completions",
            model="llama-3.1-8b-instant",  # Groq
            verbose=False,
        )

        comment = analyst.analyze(
            perf_df=perf_df,
            equities=equities,
            params={
                "strategy": "long_short",
                "hold": 10,
                "fee_buy": 0.0005,
                "fee_sell": 0.0005,
                "state_window": 60,
            },
            language="en",
        )

        print("\n===== LLM Analysis of Backtest =====\n")
        print(comment)

    except Exception as e:
        print("\n[WARN] LLM analysis failed, only an error message was printed：", e)

if __name__ == "__main__":
    main()