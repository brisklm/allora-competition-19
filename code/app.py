import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import pickle
import ast
import git

# Initialize app and env
app = Flask(__name__)
load_dotenv()

# Dynamic version tag for visibility in logs
COMPETITION = os.getenv("COMPETITION", "competition19")
TOPIC_ID = os.getenv("TOPIC_ID", "65")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 8001))

from config import *

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results.",
        "parameters": {}
    },
    {
        "name": "write_code",
        "description": "Writes complete source code to a specified file, overwriting existing content after syntax validation.",
        "parameters": {
            "title": {"type": "string", "description": "Filename (e.g., model.py)", "required": True},
            "content": {"type": "string", "description": "Complete source code content", "required": True},
            "artifact_id": {"type": "string", "description": "Artifact UUID", "required": False},
            "artifact_version_id": {"type": "string", "description": "Version UUID", "required": False},
            "contentType": {"type": "string", "description": "Content type (e.g., text/python)", "required": False}
        }
    },
    {
        "name": "commit_to_github",
        "description": "Commits changes to GitHub repository.",
        "parameters": {
            "message": {"type": "string", "description": "Commit message", "required": True},
            "files": {"type": "array", "description": "List of files to commit", "items": {"type": "string"}}
        }
    }
]

# In-memory cache for inference
MODEL_CACHE = {
    "model": None,
    "selected_features": [],
    "last_model_mtime": None,
    "last_features_mtime": None,
}

def load_model(force=False):
    model_path = model_file_path
    mtime = os.path.getmtime(model_path)
    if force or MODEL_CACHE['model'] is None or MODEL_CACHE['last_model_mtime'] != mtime:
        with open(model_path, 'rb') as f:
            MODEL_CACHE['model'] = pickle.load(f)
        MODEL_CACHE['last_model_mtime'] = mtime
    features_path = selected_features_path
    ftime = os.path.getmtime(features_path)
    if force or not MODEL_CACHE['selected_features'] or MODEL_CACHE['last_features_mtime'] != ftime:
        with open(features_path, 'r') as f:
            MODEL_CACHE['selected_features'] = json.load(f)
        MODEL_CACHE['last_features_mtime'] = ftime

def optimize():
    try:
        import optuna
    except ImportError:
        return {"error": "Optuna not available"}
    # Assume objective function in train.py tuned for R2 >0.1, directional acc >0.6, correlation >0.25
    # with params like max_depth, num_leaves, regularization, lags, momentum
    from train import objective  # Assumed to exist with optimizations
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # Increased trials
    return {"best_params": study.best_params, "best_value": study.best_value}

def write_code(params):
    title = params['title']
    content = params['content']
    try:
        ast.parse(content)
    except SyntaxError as e:
        return {"error": str(e)}
    with open(title, 'w') as f:
        f.write(content)
    return {"status": "success", "file": title}

def commit_to_github(params):
    message = params['message']
    files = params.get('files', [])
    repo = git.Repo('.')
    for f in files:
        repo.index.add(f)
    repo.index.commit(message)
    origin = repo.remote(name='origin')
    origin.push()
    return {"status": "success"}

@app.route('/invoke', methods=['POST'])
def invoke():
    data = request.json
    tool_name = data['name']
    params = data.get('parameters', {})
    if tool_name == "optimize":
        return jsonify(optimize())
    elif tool_name == "write_code":
        return jsonify(write_code(params))
    elif tool_name == "commit_to_github":
        return jsonify(commit_to_github(params))
    else:
        return jsonify({"error": "Unknown tool"})

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    features = MODEL_CACHE['selected_features']
    input_data = np.array([data.get(f, 0) for f in features]).reshape(1, -1)  # Robust default to 0
    # Robust NaN handling
    input_data = np.nan_to_num(input_data, nan=0)
    # Optional VADER sentiment if text provided
    if SentimentIntensityAnalyzer and 'text' in data:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(data['text'])['compound']
        # Assume 'vader_compound' in features, append or replace
        if 'vader_compound' in features:
            idx = features.index('vader_compound')
            input_data[0, idx] = sentiment
    pred = MODEL_CACHE['model'].predict(input_data)[0]
    # Stabilize via simple smoothing (example: average with 0, but could be ensemble)
    pred = (pred + 0) / 2  # Placeholder for ensembling
    return jsonify({"prediction": pred})

@app.route('/version', methods=['GET'])
def get_version():
    return MCP_VERSION

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)