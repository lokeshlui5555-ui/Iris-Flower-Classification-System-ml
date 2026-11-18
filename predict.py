"""
IrisAI - Inference and Prediction Module
Loads trained model and scaler to generate species predictions and probability distributions.
Can be imported as a library or executed from the command line.
"""

import os
import sys
import argparse
from unittest.mock import MagicMock

# Shim gradient boosting if Windows WDAC / AppLocker blocks unused binary extension
if 'sklearn.ensemble._gradient_boosting' not in sys.modules:
    try:
        import sklearn.ensemble._gradient_boosting
    except Exception:
        sys.modules['sklearn.ensemble._gradient_boosting'] = MagicMock()

import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Rich flower metadata and botanical descriptions
FLOWER_METADATA = {
    "Iris Setosa": {
        "scientific_name": "Iris setosa Pall. ex Link",
        "common_name": "Bristle-pointed Iris / Arctic Iris",
        "badge_class": "badge-setosa",
        "color_accent": "#3b82f6",
        "image_file": "setosa.svg",
        "description": "Iris Setosa is characterized by strikingly small, narrow petals (1.0 - 1.9 cm) and wide sepals. It forms a clean, linearly separable cluster in dimensional feature space.",
        "distinct_traits": "Petal length < 2.5 cm, broad sepal width, and crisp violet-blue perianth with delicate dark veining.",
        "typical_habitat": "Boreal marshes, coastal dunes, and arctic tundra meadows across Alaska, Canada, and Northeast Asia.",
        "typical_dimensions": {
            "sepal_length": "4.3 - 5.8 cm",
            "sepal_width": "2.3 - 4.4 cm",
            "petal_length": "1.0 - 1.9 cm",
            "petal_width": "0.1 - 0.6 cm"
        }
    },
    "Iris Versicolor": {
        "scientific_name": "Iris versicolor L.",
        "common_name": "Blue Flag / Harlequin Blueflag",
        "badge_class": "badge-versicolor",
        "color_accent": "#10b981",
        "image_file": "versicolor.svg",
        "description": "Iris Versicolor exhibits intermediate petal lengths and widths. It displays deep blue-violet blossoms with yellow-green signal splashes and dark purple venation.",
        "distinct_traits": "Moderate petal length (3.0 - 5.1 cm) and petal width (1.0 - 1.8 cm), serving as an evolutionary bridge species.",
        "typical_habitat": "Wetlands, stream banks, freshwater marshes, and sedge meadows throughout eastern North America.",
        "typical_dimensions": {
            "sepal_length": "4.9 - 7.0 cm",
            "sepal_width": "2.0 - 3.4 cm",
            "petal_length": "3.0 - 5.1 cm",
            "petal_width": "1.0 - 1.8 cm"
        }
    },
    "Iris Virginica": {
        "scientific_name": "Iris virginica L.",
        "common_name": "Virginia Iris / Great Blue Flag",
        "badge_class": "badge-virginica",
        "color_accent": "#8b5cf6",
        "image_file": "virginica.svg",
        "description": "Iris Virginica possesses the largest and most robust petals in the Iris family, with prominent petal widths up to 2.5 cm and large, stately violet-purple blooms.",
        "distinct_traits": "Petal length > 4.5 cm and petal width > 1.4 cm, distinguished by bright pubescent yellow crests on its falls.",
        "typical_habitat": "Lowland swamps, river floodplains, and coastal freshwater retention basins in the southeastern United States.",
        "typical_dimensions": {
            "sepal_length": "4.9 - 7.9 cm",
            "sepal_width": "2.2 - 3.8 cm",
            "petal_length": "4.5 - 6.9 cm",
            "petal_width": "1.4 - 2.5 cm"
        }
    }
}


def load_artifacts():
    """Load trained best model and scaler artifact."""
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Model artifacts not found in {MODELS_DIR}. Please run train_model.py first."
        )

    bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return bundle, scaler


def validate_inputs(sepal_length, sepal_width, petal_length, petal_width):
    """Validate that measurements are positive floats within reasonable botanical bounds."""
    try:
        sl = float(sepal_length)
        sw = float(sepal_width)
        pl = float(petal_length)
        pw = float(petal_width)
    except (ValueError, TypeError):
        return False, "All measurements must be valid numeric values.", None

    inputs = [('Sepal Length', sl), ('Sepal Width', sw), ('Petal Length', pl), ('Petal Width', pw)]
    
    for name, val in inputs:
        if val <= 0:
            return False, f"{name} must be greater than 0 cm.", None
        if val > 15.0:
            return False, f"{name} ({val} cm) exceeds realistic botanical dimensions (max 15.0 cm).", None

    return True, "", [sl, sw, pl, pw]


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Perform flower species prediction using the trained model.
    Returns structured prediction dictionary with class probabilities and metadata.
    """
    valid, err_msg, features = validate_inputs(sepal_length, sepal_width, petal_length, petal_width)
    if not valid:
        return {"success": False, "error": err_msg}

    bundle, scaler = load_artifacts()
    clf = bundle["model"]
    use_scaled = bundle.get("use_scaled", False)
    species_names = bundle.get("species_names", ["Iris Setosa", "Iris Versicolor", "Iris Virginica"])

    raw_features = np.array([features], dtype=np.float64)
    model_features = scaler.transform(raw_features) if use_scaled else raw_features

    # Predict class index
    pred_idx = int(clf.predict(model_features)[0])
    predicted_species = species_names[pred_idx]

    # Predict class probabilities
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(model_features)[0]
    else:
        # Fallback for models without predict_proba
        probs = np.zeros(len(species_names))
        probs[pred_idx] = 1.0

    probabilities = {
        name: round(float(prob) * 100, 2)
        for name, prob in zip(species_names, probs)
    }
    confidence = probabilities[predicted_species]

    flower_info = FLOWER_METADATA.get(predicted_species, {})

    return {
        "success": True,
        "predicted_species": predicted_species,
        "species_id": pred_idx,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_name": bundle.get("name", "Best Model"),
        "features": {
            "sepal_length": features[0],
            "sepal_width": features[1],
            "petal_length": features[2],
            "petal_width": features[3]
        },
        "metadata": flower_info
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IrisAI Flower Species Predictor")
    parser.add_argument("--sepal-length", "-sl", type=float, default=5.1, help="Sepal length in cm")
    parser.add_argument("--sepal-width", "-sw", type=float, default=3.5, help="Sepal width in cm")
    parser.add_argument("--petal-length", "-pl", type=float, default=1.4, help="Petal length in cm")
    parser.add_argument("--petal-width", "-pw", type=float, default=0.2, help="Petal width in cm")

    args = parser.parse_args()

    print(f"\nEvaluating Measurements:")
    print(f"  Sepal: {args.sepal_length} x {args.sepal_width} cm")
    print(f"  Petal: {args.petal_length} x {args.petal_width} cm\n")

    res = predict_species(args.sepal_length, args.sepal_width, args.petal_length, args.petal_width)

    if res["success"]:
        print(f"Predicted Species: {res['predicted_species']}")
        print(f"Confidence       : {res['confidence']}%")
        print(f"Model Employed   : {res['model_name']}")
        print("Class Probabilities:")
        for sp, p in res["probabilities"].items():
            print(f"  - {sp:16s}: {p:.2f}%")
    else:
        print(f"Error: {res['error']}")
