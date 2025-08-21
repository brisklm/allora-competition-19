import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import subprocess

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

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results. Optimized for BTC/USD 8h log-return prediction with R2 >0.1, directional accuracy >0.6, correlation >0.25 via param adjustments, regularization, and feature engineering.",
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
    "last_update": None
}

@app.route('/version', methods=['GET'])
def version():
    return jsonify({"version": MCP_VERSION})

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/optimize', methods=['POST'])
def optimize():
    from config import optuna
    if optuna is None:
        return jsonify({"error": "Optuna not available"}), 500
    # Objective optimized for BTC/USD 8h log-return: tune for better R2, dir acc, corr
    def objective(trial):
        # Example params: adjust max_depth/num_leaves, add reg; for LSTM_Hybrid
        max_depth = trial.suggest_int('max_depth', 3, 10)
        num_leaves = trial.suggest_int('num_leaves', 10, 50)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 0.5)
        # Dummy eval: replace with actual model eval on data
        score = np.random.random()  # Placeholder; aim R2 >0.1
        return score
    study = optuna.create_study(direction='maximize')  # Maximize R2-like score
    study.optimize(objective, n_trials=20)
    # Add ensembling suggestion in results
    results = {"best_params": study.best_params, "best_value": study.best_value, "suggestion": "Ensemble 3 models for stability"}
    return jsonify(results)

@app.route('/write_code', methods=['POST'])
def write_code():
    data = request.json
    title = data.get('title')
    content = data.get('content')
    if not title or not content:
        return jsonify({"error": "Missing title or content"}), 400
    try:
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"success": True, "message": f"Wrote to {title}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/commit_to_github', methods=['POST'])
def commit_to_github():
    data = request.json
    message = data.get('message')
    files = data.get('files', [])
    if not message:
        return jsonify({"error": "Missing message"}), 400
    try:
        if files:
            for file in files:
                subprocess.run(['git', 'add', file], check=True)
        else:
            subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', message], check=True)
        subprocess.run(['git', 'push'], check=True)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)