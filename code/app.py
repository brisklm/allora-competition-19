import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np

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
        "description": "Triggers model optimization using Optuna tuning for improved R2 >0.1, directional accuracy >0.6, with regularization and ensembling. Returns results.",
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
MODEL_CACHE = {"model": None, "selected_features": []}

@app.route("/inference/<token>", methods=["GET"])
def inference(token):
    if token.upper() != TOKEN:
        return jsonify({"error": "Invalid token"}), 400
    # Load model if not cached
    if MODEL_CACHE["model"] is None:
        try:
            from config import model_file_path, selected_features_path
            import pickle
            with open(model_file_path, 'rb') as f:
                MODEL_CACHE["model"] = pickle.load(f)
            with open(selected_features_path, 'r') as f:
                MODEL_CACHE["selected_features"] = json.load(f)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    # For demonstration, generate dummy input with selected features
    # Add low-variance check and NaN handling
    selected_features = MODEL_CACHE["selected_features"]
    input_data = np.random.randn(1, len(selected_features))  # Dummy
    # Check for low variance
    variances = np.var(input_data, axis=0)
    low_var_mask = variances < 1e-6
    if np.any(low_var_mask):
        print("Low variance features detected.")
    # Check for NaNs
    if np.any(np.isnan(input_data)):
        input_data = np.nan_to_num(input_data)
    # Predict
    try:
        prediction = MODEL_CACHE["model"].predict(input_data)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"prediction": prediction})

@app.route("/inference", methods=["POST"])
def inference_post():
    data = request.json
    token = data.get("token")
    features = data.get("features")
    if token.upper() != TOKEN or not features:
        return jsonify({"error": "Invalid input"}), 400
    input_data = np.array([features])[None, :]  # shape (1, n_features)
    input_data = np.nan_to_num(input_data)
    prediction = MODEL_CACHE["model"].predict(input_data)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=True)