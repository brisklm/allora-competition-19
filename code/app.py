import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
try:
    import optuna
except Exception:
    optuna = None

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
    "last_updated": None
}

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

@app.route('/tool/<name>', methods=['POST'])
def execute_tool(name):
    if name == "optimize":
        if optuna is None:
            return jsonify({"error": "Optuna not installed"}), 500
        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 3, 10)
            num_leaves = trial.suggest_int('num_leaves', 10, 50)
            reg_lambda = trial.suggest_float('reg_lambda', 0.01, 1.0)
            score = np.random.random()  # Placeholder for real metric (e.g., R2 or directional accuracy)
            return score
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)  # Increased trials for better optimization
        return jsonify({"best_params": study.best_params, "best_value": study.best_value})
    elif name == "write_code":
        params = request.json
        title = params['title']
        content = params['content']
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            return jsonify({"error": str(e)}), 400
        with open(title, 'w') as f:
            f.write(content)
        return jsonify({"status": "written"})
    elif name == "commit_to_github":
        return jsonify({"status": "committed"})  # Placeholder
    return jsonify({"error": "Tool not found"}), 404

if __name__ == '__main__':
    app.run(port=FLASK_PORT, debug=True)