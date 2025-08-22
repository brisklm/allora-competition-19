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
        "description": "Triggers model optimization using Optuna tuning with adjustments to max_depth/num_leaves, adding regularization, to improve R2 above 0.1, increase directional accuracy above 0.6, engineer sign/log-return lags and momentum filters, increase correlation magnitude above 0.25, stabilize predictions via smoothing or ensembling, incorporate VADER sentiment, ensure robust NaN handling and low-variance checks, and returns results.",
        "parameters": {}
    },
    {
        "name": "write_code",
        "description": "Writes complete source code to a specified file, overwriting existing content after syntax validation.",
        "parameters": {
            "title": {"type": "string", "description": "Filename (e.g., model.py)", "required": true},
            "content": {"type": "string", "description": "Complete source code content", "required": true},
            "artifact_id": {"type": "string", "description": "Artifact UUID", "required": false},
            "artifact_version_id": {"type": "string", "description": "Version UUID", "required": false},
            "contentType": {"type": "string", "description": "Content type (e.g., text/python)", "required": false}
        }
    },
    {
        "name": "commit_to_github",
        "description": "Commits changes to GitHub repository.",
        "parameters": {
            "commit_message": {"type": "string", "description": "The commit message", "required": true},
            "files": {"type": "array", "description": "List of files to commit", "required": false}
        }
    }
]

@app.route('/', methods=['GET'])
def home():
    return f"MCP Version: {MCP_VERSION}"

@app.route('/tools', methods=['GET'])
def get_tools():
    return jsonify(TOOLS)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)