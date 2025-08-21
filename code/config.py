import os
from datetime import datetime
import numpy as np
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None
try:
    import optuna
except Exception:
    optuna = None
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
SELECTED_FEATURES = ['volatility_BTCUSDT', 'volume_change_BTCUSDT', 'momentum_BTCUSDT', 'rsi_BTCUSDT', 'ma5_BTCUSDT', 'ma20_BTCUSDT', 'macd_BTCUSDT', 'bb_upper_BTCUSDT', 'bb_lower_BTCUSDT', 'sign_log_return_lag1_BTCUSDT', 'garch_vol_BTCUSDT', 'volatility_ETHUSDT', 'volume_change_ETHUSDT', 'sol_btc_corr', 'sol_eth_corr', 'sol_btc_vol_ratio', 'sol_btc_volume_ratio', 'sol_eth_vol_ratio', 'sol_eth_momentum_ratio', 'sentiment_score', 'sol_tx_volume', 'hour_of_day', *[f'open_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'high_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'low_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'close_ETHUSDT_lag{i}' for i in range(1, 11)], *[f'open_BTCUSDT_lag{i}' for i in range(1, 11)], *[f'high_BTCUSDT_lag{i}' for i in range(1, 11)], *[f'low_BTCUSDT_lag{i}' for i in range(1, 11)], *[f'close_BTCUSDT_lag{i}' for i in range(1, 11)]]
MODEL_PARAMS = {'n_estimators': 200, 'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': -1, 'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8, 'n_jobs': 1, 'hidden_size': 64, 'num_layers': 2}
OPTUNA_TRIALS = int(os.getenv('OPTUNA_TRIALS', 50))
USE_SYNTHETIC_DATA = os.getenv('USE_SYNTHETIC_DATA', 'True').lower() == 'true'