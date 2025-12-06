# Alternative Data Quant Pipeline

This repository implements a modular quantitative research and backtesting system that integrates:
- Market data (Yahoo Finance)
- Alternative data (Google Trends)
- Feature engineering
- Machine Learning models (Regression, Classification, LSTM)
- Reinforcement Learning (CQL)
- Unified backtesting framework
- Optional LLM modules for automated hyperparameter tuning and result interpretation

The goal is to provide a clean, extensible pipeline where each file has a single responsibility
and components can be replaced or upgraded independently.

---

## File Overview

### **Data_handler.py**
Handles all data acquisition and preprocessing:
- `MarketDataHandler`:  
  Downloads historical prices using Yahoo Finance, cleans missing data, computes returns.
- `GoogleTrendHandler`:  
  Fetches search interest time series via Google Trends API, fills missing values.
- `DataMerger`:  
  Aligns market data and trend data, constructs engineered features  
  (momentum/volatility for each ticker, momentum for each trend keyword).  
  Saves `merged_features_multiasset.xlsx`.

---

### **Integrate_class.py**
The central training pipeline that coordinates all model families:
- Loads merged features  
- Standardizes inputs  
- Prepares multi-asset targets (next-day returns)  
- Trains:
  - Regression models (TriRegressorTrainer)
  - LSTM regressors
  - Classification models (TriClassifierTrainer)
  - Reinforcement learning policy (CQLPolicyTrainer)
- Optionally uses `LLMHyperParamAssistant` to auto-generate training hyperparameters.
- Returns a dictionary of trained models:  
  `{"pred": [...], "cls": [...], "rl": [...]}`

This is the main entry point used before backtesting.

---

### **TriRegressorTrainer.py**
Implements a unified interface for training **three regression models**:
- Linear Regression  
- Random Forest Regressor  
- Bayesian Ridge  

Supports **multi-output regression** (multiple tickers).  
Automatically filters invalid hyperparameters and ensures reproducibility.

---

### **Classifier_class.py**
Trainer for classification-based prediction:
- XGBoost Classifier  
- Random Forest Classifier  

Uses binary labels based on whether the next return is positive.  
Returns a list of trained classifier models.

---

### **LSTM_class.py**
Provides a lightweight PyTorch implementation of:
- Two-layer LSTM network  
- Supports multiple output heads (one per asset)  
- Uses Adam optimizer and MSE loss  
- Wrapped with a `.predict()` function to match scikit-learn style

---

### **CQL_class.py**
Implements Conservative Q-Learning for trading:
- Builds an MDP dataset from returns
- Defines actions as {sell, hold, buy}
- Conservative Q-learning avoids overestimation
- Trains using `d3rlpy`  
- Returns a wrapped policy with `.predict()` producing discrete actions

---

### **LLM.py (LLMHyperParamAssistant)**
Optional module that:
- Reads selected code files as “context”
- Accepts natural-language training objectives (e.g., “avoid overfitting”)
- Calls an external LLM (OpenAI-compatible API)
- Produces a structured JSON that configures all trainers:
  - LSTM hyperparameters
  - Regression model params
  - Classifier params
  - RL training params

This makes the pipeline adaptive without manual tuning.

---

### **backtest.py**
A unified engine for evaluating all model types under a consistent framework:
- `default_feature_maker`: loads engineered features for backtesting
- `ModelAdapter`: standardizes predictors/classifiers/RL into a single `.infer()` interface
- Signal generation:
  - predictor → continuous signal  
  - classifier → long/short  
  - RL → discrete actions
- Position logic:
  - `backtest_hold_H()` for ML strategies
  - `backtest_daily_actions()` for RL policies
- Returns:
  - return table
  - equity curves
  - position weights
- Includes basic performance metrics (CAGR, Sharpe, MaxDD, AnnRet, AnnVol)

---

### **backtest_llm.py**
Optional report generator:
- Summarizes performance tables and equity curves
- Calls an external LLM API to convert numerical results into a readable analysis
- Supports English and Chinese output modes

---

### **integrated_to_backtest.py**
Example end-to-end script that:
1. Downloads market data & Google Trends  
2. Merges all features  
3. Runs the integrated training pipeline  
4. Assigns the model set to each ticker  
5. Runs backtests  
6. Prints performance tables  
7. (Optional) Uses an LLM to generate commentary  

This script demonstrates the intended workflow of the entire system.

---

### **merged_features_multiasset.xlsx / returns_data.xlsx / googletrend_data.xlsx**
Generated artifacts that store intermediate data:
- Engineered multi-asset feature matrix  
- Daily returns  
- Google Trends time series  

These are loaded automatically during training and backtesting.

---

##  Dependencies
- Python 3.10+
- pandas, numpy  
- scikit-learn  
- PyTorch  
- d3rlpy  
- yfinance  
- pytrends  
- Any OpenAI-compatible LLM API (optional)

---

## Typical Usage

Train all models:
```python
integrator = Integrated()
model_map = integrator.return_trend(returns, trends, tickers)
