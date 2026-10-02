import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import FEATURES, validate_input

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "artifacts" / "student_performance_pipeline.joblib"
META_PATH = BASE / "artifacts" / "metadata.json"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
INTERVAL = metadata["prediction_interval"]


def performance_label(score):
    if score >= 90:
        return "Outstanding"
    if score >= 80:
        return "Excellent"
    if score >= 70:
        return "Very Good"
    if score >= 60:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"


def predict(raw_row: dict) -> dict:
    """Validate input, predict, clip to 0-100 and attach a 90% prediction interval."""
    row = validate_input(raw_row)
    score = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
    score = max(0.0, min(100.0, score))
    return {
        "prediction": round(score, 1),
        "label": performance_label(score),
        "interval_low": round(max(0.0, score + INTERVAL["lower"]), 1),
        "interval_high": round(min(100.0, score + INTERVAL["upper"]), 1),
        "interval_coverage": INTERVAL["coverage"],
        "at_risk": score < 50,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result, error = None, None
    if request.method == "POST":
        try:
            result = predict({col: request.form.get(col) for col in FEATURES})
        except ValueError as e:
            error = str(e)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Please check the inputs and try again."
    return render_template(
        "index.html",
        result=result,
        prediction=result["prediction"] if result else None,
        label=result["label"] if result else None,
        error=error,
        model_name=metadata["best_model"],
        test_r2=metadata["test_metrics"]["r2"],
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object"), 400
    try:
        return jsonify(predict(payload))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.route("/health")
def health():
    return jsonify(status="ok", model=metadata["best_model"])


if __name__ == "__main__":
    # debug is off by default; use `FLASK_DEBUG=1 python app.py` locally if needed
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
