"""
IrisAI - Intelligent Iris Flower Classification System
Flask Backend Application
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from predict import predict_species, FLOWER_METADATA

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')


def load_cached_metrics():
    """Load precomputed metrics and dataset stats from JSON."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            app.logger.error(f"Error loading metrics.json: {e}")
    return {}


@app.route('/')
def index():
    """Home page."""
    metrics = load_cached_metrics()
    return render_template('index.html', metrics=metrics)


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    """
    Prediction page and endpoint.
    GET: renders prediction.html
    POST (form-encoded or JSON): returns prediction result.
    """
    if request.method == 'POST':
        # Accept either JSON body or standard form POST
        data = request.get_json(silent=True) or request.form.to_dict()
        try:
            sl = float(data.get('sepal_length', 0))
            sw = float(data.get('sepal_width', 0))
            pl = float(data.get('petal_length', 0))
            pw = float(data.get('petal_width', 0))
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": "All 4 measurements must be valid numeric values."
            }), 400

        result = predict_species(sl, sw, pl, pw)
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code

    metrics = load_cached_metrics()
    best_model_name = metrics.get('best_model', 'Random Forest')
    return render_template('prediction.html', best_model_name=best_model_name)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Dedicated JSON REST API endpoint for flower classification.
    Expects:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error": "Missing JSON request payload."
        }), 400

    required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for field in required_fields:
        if field not in data:
            return jsonify({
                "success": False,
                "error": f"Missing required feature: '{field}'."
            }), 400

    try:
        sl = float(data['sepal_length'])
        sw = float(data['sepal_width'])
        pl = float(data['petal_length'])
        pw = float(data['petal_width'])
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "All measurements must be numeric floats or integers."
        }), 400

    result = predict_species(sl, sw, pl, pw)
    status_code = 200 if result.get('success') else 400
    return jsonify(result), status_code


@app.route('/analytics')
def analytics():
    """Analytics dashboard page displaying performance comparison and EDA charts."""
    metrics = load_cached_metrics()
    best_model_name = metrics.get('best_model', 'Random Forest')
    return render_template('analytics.html', metrics=metrics, best_model_name=best_model_name)


@app.route('/api/metrics')
def api_metrics():
    """JSON API endpoint returning model metrics for client-side rendering."""
    metrics = load_cached_metrics()
    return jsonify(metrics)


@app.route('/about')
def about():
    """About page with methodology, algorithm profiles, and dataset history."""
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404


if __name__ == '__main__':
    # Default local dev port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
