import os
from datetime import datetime
import numpy as np
# Optional: avoid hard dependency on nltk in runtime
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore
try:
    import optuna  # noqa: F401
except Exception:  # pragma: no cover
    optuna = None  # type: ignore
data_base_path = os.path.join(os.getcwd(), 'data')
model_file_path = os.path.join(data_base_path, 'model.pkl')
scaler_file_path = os.path.join(data_base_path, 'scaler.pkl')
training_price_data_path = os.path.join(data_base_path, 'price_data.csv')
selected_features_path = os.path.join(data_base_path, 'selected_features.json')
best_model_info_path = os.path.join(data_base_path, 'best_model.json')
sol_source_path = os.path.join(data_base_path, os.getenv('SOL_SOURCE', 'raw_sol.csv'))
eth_source_path = os.path.join(data_base_path, os.getenv('ETH_SOURCE', 'raw_eth.csv'))
features_sol_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH', 'features_sol.csv'))
features_eth_path = os.path.join(data_base_path, os.getenv('FEATURES_PATH_ETH', 'features_eth.csv'))
# Competition 19: BTC/USD 8h log-return prediction (5min updates)
TOKEN = os.getenv('TOKEN', 'BTC')
TIMEFRAME = os.getenv('TIMEFRAME', '8h')
TRAINING_DAYS = int(os.getenv('TRAINING_DAYS', 365))
MINIMUM_DAYS = 180
REGION = os.getenv('REGION', 'com')
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'binance')
MODEL = os.getenv('MODEL', 'LSTM_Hybrid')
CG_API_KEY = os.getenv('CG_API_KEY', 'CG-xA5NyokGEVbc4bwrvJPcpZvT')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY', '70ed65ce-4750-4fd5-83bd-5aee9aa79ead')
HELIUS_RPC_URL = os.getenv('HELIUS_RPC_URL', 'https://mainnet.helius-rpc.com')
BITQUERY_API_KEY = os.getenv('BITQUERY_API_KEY', 'ory_at_LmFLzUutMY8EVb-P_PQVP9ntfwUVTV05LMal7xUqb2I.vxFLfMEoLGcu4XoVi47j-E2bspraTSrmYzCt1A4y2k')
# Feature set adapted to BTC/USD 8h log-return prediction (Competition 19, topic 65)
# Keep only features that our pipeline can handle, add lags, momentum, sentiment for optimization
FEATURES = [
    'close', 'volume', 'rsi_14', 'sma_50', 'ema_20',
    'log_return', 'log_return_lag1', 'log_return_lag2', 'log_return_lag3',
    'sign_log_return', 'momentum_5', 'momentum_10',
    'vader_sentiment_compound' if SentimentIntensityAnalyzer else None,
]
# Filter None
FEATURES = [f for f in FEATURES if f is not None]
# For model optimization with Optuna
OPTUNA_TRIALS = 100  # Increased for better tuning
MODEL_PARAMS = {
    'max_depth': 8,  # Adjusted
    'num_leaves': 25,  # Adjusted
    'reg_alpha': 0.05,  # Added regularization
    'reg_lambda': 0.05,  # Added regularization
}
# For NaN handling
IMPUTE_METHOD = 'ffill'  # Robust forward fill for time series
# Low variance threshold
VARIANCE_THRESHOLD = 0.005  # Stricter for low-variance checks
# For stabilizing predictions
SMOOTHING_WINDOW = 3  # For smoothing
ENSEMBLE = True  # Enable ensembling for stability
ENSEMBLE_MODELS = ['LSTM_Hybrid', 'LightGBM']  # Ensemble for better accuracy