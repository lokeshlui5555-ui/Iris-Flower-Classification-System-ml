"""
IrisAI - Machine Learning Training Pipeline
Trains, evaluates, and compares 4 classification algorithms on the Iris Dataset:
1. Decision Tree Classifier
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)
4. Logistic Regression
"""

import os
import sys
import json
import csv
from unittest.mock import MagicMock

# Shim gradient boosting if Windows WDAC / AppLocker blocks unused binary extension
if 'sklearn.ensemble._gradient_boosting' not in sys.modules:
    try:
        import sklearn.ensemble._gradient_boosting
    except Exception:
        sys.modules['sklearn.ensemble._gradient_boosting'] = MagicMock()

import numpy as np
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'images')

for d in [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, STATIC_IMG_DIR]:
    os.makedirs(d, exist_ok=True)


def export_dataset_csv(iris):
    """Save the Iris dataset to CSV format for portability."""
    csv_path = os.path.join(DATASET_DIR, 'iris.csv')
    headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'species_id']
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(iris.data)):
            row = list(iris.data[i]) + [iris.target_names[iris.target[i]], int(iris.target[i])]
            writer.writerow(row)
            
    print(f"[✓] Dataset exported to: {csv_path}")
    return csv_path


def train_and_evaluate():
    print("=" * 60)
    print("IrisAI Model Training & Benchmark Suite")
    print("=" * 60)

    # 1. Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
    species_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

    export_dataset_csv(iris)

    # 2. Train-Test Split (80% Train, 20% Test, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"[✓] Feature scaler saved to: {scaler_path}")

    # 4. Define Candidate Models
    models = {
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=4, random_state=42, criterion='gini'),
            "use_scaled": False,
            "description": "Tree-based recursive binary splitting model with transparent decision boundaries."
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
            "use_scaled": False,
            "description": "Ensemble of 100 decision trees combining bagging and random feature subspaces."
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2),
            "use_scaled": True,
            "description": "Instance-based non-parametric classifier using Euclidean distance metric."
        },
        "Logistic Regression": {
            "model": LogisticRegression(C=1.0, max_iter=200, random_state=42),
            "use_scaled": True,
            "description": "Multinomial linear model with softmax probability distribution."
        }
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = None
    best_score = -1.0

    for name, config in models.items():
        print(f"\n--- Training {name} ---")
        clf = config["model"]
        X_tr = X_train_scaled if config["use_scaled"] else X_train
        X_te = X_test_scaled if config["use_scaled"] else X_test
        X_full = scaler.transform(X) if config["use_scaled"] else X

        # Fit model
        clf.fit(X_tr, y_train)

        # Predictions
        y_train_pred = clf.predict(X_tr)
        y_test_pred = clf.predict(X_te)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # 5-Fold Cross Validation
        cv_scores = cross_val_score(clf, X_full, y, cv=cv, scoring='accuracy')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_test_pred).tolist()
        report = classification_report(y_test, y_test_pred, target_names=species_names, output_dict=True)

        print(f"Train Accuracy: {train_acc * 100:.2f}%")
        print(f"Test Accuracy : {test_acc * 100:.2f}%")
        print(f"5-Fold CV Mean: {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
        print(f"F1 Score      : {f1:.4f}")

        results[name] = {
            "name": name,
            "description": config["description"],
            "use_scaled": config["use_scaled"],
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "cv_scores": [round(float(s), 4) for s in cv_scores],
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "classification_report": report
        }

        # Save individual model
        model_filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(clf, os.path.join(MODELS_DIR, model_filename))

        # Best model criterion: higher CV mean, tiebreak on test accuracy
        overall_score = cv_mean * 0.7 + test_acc * 0.3
        if overall_score > best_score:
            best_score = overall_score
            best_model_name = name

    print(f"\n[★] Best performing model: {best_model_name} (Combined Score: {best_score:.4f})")

    # Save best model to models/best_model.joblib
    best_clf = models[best_model_name]["model"]
    best_model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump({
        "name": best_model_name,
        "model": best_clf,
        "use_scaled": models[best_model_name]["use_scaled"],
        "species_names": species_names,
        "feature_names": feature_names
    }, best_model_path)
    print(f"[✓] Best model artifact saved to: {best_model_path}")

    # Dataset Summary Statistics for Analytics
    dataset_stats = {
        "total_samples": len(X),
        "total_features": len(feature_names),
        "total_species": len(species_names),
        "species_names": species_names,
        "feature_names": feature_names,
        "species_counts": {name: int(np.sum(y == i)) for i, name in enumerate(species_names)},
        "feature_summary": {}
    }

    for idx, f_name in enumerate(feature_names):
        vals = X[:, idx]
        species_breakdown = {}
        for sp_idx, sp_name in enumerate(species_names):
            sp_vals = X[y == sp_idx, idx]
            species_breakdown[sp_name] = {
                "min": round(float(np.min(sp_vals)), 2),
                "mean": round(float(np.mean(sp_vals)), 2),
                "max": round(float(np.max(sp_vals)), 2),
                "std": round(float(np.std(sp_vals)), 2)
            }
        dataset_stats["feature_summary"][f_name] = {
            "overall_min": round(float(np.min(vals)), 2),
            "overall_mean": round(float(np.mean(vals)), 2),
            "overall_max": round(float(np.max(vals)), 2),
            "overall_std": round(float(np.std(vals)), 2),
            "by_species": species_breakdown
        }

    # Complete Metrics JSON
    payload = {
        "best_model": best_model_name,
        "best_model_metrics": results[best_model_name],
        "models": results,
        "dataset_stats": dataset_stats
    }

    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"[✓] Benchmark metrics saved to: {metrics_path}")

    # Generate visual artifacts
    generate_charts(payload, X, y, species_names, feature_names)

    return payload


def generate_charts(payload, X, y, species_names, feature_names):
    """
    Generate clean, vector-crisp standalone visual charts in SVG format
    and attempt PNG generation if matplotlib is available.
    """
    models = payload["models"]
    names = list(models.keys())
    test_accs = [models[m]["test_accuracy"] * 100 for m in names]
    cv_means = [models[m]["cv_mean"] * 100 for m in names]

    # 1. Model Comparison Chart (SVG)
    svg_bar = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 380" width="100%" height="100%">
    <defs>
        <linearGradient id="gradBar1" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#8b5cf6"/>
            <stop offset="100%" stop-color="#6d28d9"/>
        </linearGradient>
        <linearGradient id="gradBar2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#c084fc"/>
            <stop offset="100%" stop-color="#a855f7"/>
        </linearGradient>
    </defs>
    <rect width="760" height="380" fill="#ffffff" rx="16"/>
    <text x="380" y="36" text-anchor="middle" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="#1e1b4b">
        Algorithm Performance Comparison
    </text>
    <text x="380" y="58" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#6b7280">
        Test Accuracy vs 5-Fold Cross-Validation Score
    </text>

    <!-- Legend -->
    <rect x="250" y="75" width="14" height="14" rx="3" fill="url(#gradBar1)"/>
    <text x="272" y="87" font-family="system-ui, sans-serif" font-size="12" fill="#374151">Test Accuracy</text>
    <rect x="410" y="75" width="14" height="14" rx="3" fill="url(#gradBar2)"/>
    <text x="432" y="87" font-family="system-ui, sans-serif" font-size="12" fill="#374151">5-Fold CV Mean</text>

    <!-- Grid lines -->
    <line x1="80" y1="280" x2="700" y2="280" stroke="#e5e7eb" stroke-width="1"/>
    <line x1="80" y1="235" x2="700" y2="235" stroke="#f3f4f6" stroke-width="1"/>
    <line x1="80" y1="190" x2="700" y2="190" stroke="#f3f4f6" stroke-width="1"/>
    <line x1="80" y1="145" x2="700" y2="145" stroke="#f3f4f6" stroke-width="1"/>
    <line x1="80" y1="100" x2="700" y2="100" stroke="#f3f4f6" stroke-width="1"/>

    <text x="70" y="284" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#9ca3af">80%</text>
    <text x="70" y="239" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#9ca3af">85%</text>
    <text x="70" y="194" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#9ca3af">90%</text>
    <text x="70" y="149" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#9ca3af">95%</text>
    <text x="70" y="104" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" fill="#9ca3af">100%</text>
"""
    # Plot bars: range is 80% to 100% -> y goes from 280 to 100 (height 180 = 20%)
    for i, m_name in enumerate(names):
        bx = 120 + i * 145
        t_acc = test_accs[i]
        c_acc = cv_means[i]

        h1 = max(10, ((t_acc - 80) / 20) * 180)
        y1 = 280 - h1
        h2 = max(10, ((c_acc - 80) / 20) * 180)
        y2 = 280 - h2

        svg_bar += f"""
        <g>
            <rect x="{bx}" y="{y1:.1f}" width="32" height="{h1:.1f}" rx="5" fill="url(#gradBar1)"/>
            <text x="{bx + 16}" y="{y1 - 6:.1f}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#6d28d9">{t_acc:.1f}%</text>

            <rect x="{bx + 38}" y="{y2:.1f}" width="32" height="{h2:.1f}" rx="5" fill="url(#gradBar2)"/>
            <text x="{bx + 54}" y="{y2 - 6:.1f}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#9333ea">{c_acc:.1f}%</text>

            <text x="{bx + 35}" y="308" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" font-weight="500" fill="#374151">{m_name}</text>
        </g>
        """

    svg_bar += "\n</svg>"

    with open(os.path.join(OUTPUTS_DIR, 'model_comparison.svg'), 'w', encoding='utf-8') as f:
        f.write(svg_bar)
    with open(os.path.join(STATIC_IMG_DIR, 'model_comparison.svg'), 'w', encoding='utf-8') as f:
        f.write(svg_bar)

    # 2. Confusion Matrix Heatmap (SVG)
    best_cm = payload["best_model_metrics"]["confusion_matrix"]
    best_name = payload["best_model"]
    cm_max = max(max(row) for row in best_cm)

    svg_cm = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 420" width="100%" height="100%">
    <rect width="540" height="420" fill="#ffffff" rx="16"/>
    <text x="270" y="32" text-anchor="middle" font-family="system-ui, sans-serif" font-size="17" font-weight="700" fill="#1e1b4b">
        Confusion Matrix ({best_name})
    </text>
    <text x="270" y="52" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#6b7280">
        Actual vs Predicted Classes on Test Split (30 Samples)
    </text>

    <!-- Axis Labels -->
    <text x="295" y="85" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="#4b5563">PREDICTED SPECIES</text>
    <text x="20" y="245" transform="rotate(-90 20 245)" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" font-weight="600" fill="#4b5563">ACTUAL SPECIES</text>
    """

    for j, sp in enumerate(["Setosa", "Versicolor", "Virginica"]):
        svg_cm += f'<text x="{190 + j * 105}" y="112" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#6d28d9">{sp}</text>\n'
        svg_cm += f'<text x="130" y="{170 + j * 90}" text-anchor="end" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#6d28d9">{sp}</text>\n'

    for r in range(3):
        for c in range(3):
            val = best_cm[r][c]
            intensity = val / max(1, cm_max)
            bg = f"rgba(109, 40, 217, {0.10 + intensity * 0.85:.2f})" if val > 0 else "#f9fafb"
            text_color = "#ffffff" if intensity > 0.45 else "#374151"
            cell_x = 145 + c * 105
            cell_y = 125 + r * 90

            svg_cm += f"""
            <rect x="{cell_x}" y="{cell_y}" width="90" height="75" rx="8" fill="{bg}" stroke="#ede9fe" stroke-width="1.5"/>
            <text x="{cell_x + 45}" y="{cell_y + 44}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="20" font-weight="700" fill="{text_color}">{val}</text>
            """

    svg_cm += "\n</svg>"

    with open(os.path.join(OUTPUTS_DIR, 'confusion_matrix.svg'), 'w', encoding='utf-8') as f:
        f.write(svg_cm)
    with open(os.path.join(STATIC_IMG_DIR, 'confusion_matrix.svg'), 'w', encoding='utf-8') as f:
        f.write(svg_cm)

    # 3. Feature Distributions Overview (SVG)
    svg_feat = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 380" width="100%" height="100%">
    <rect width="760" height="380" fill="#ffffff" rx="16"/>
    <text x="380" y="32" text-anchor="middle" font-family="system-ui, sans-serif" font-size="18" font-weight="700" fill="#1e1b4b">
        Iris Feature Means by Species
    </text>
    <text x="380" y="52" text-anchor="middle" font-family="system-ui, sans-serif" font-size="12" fill="#6b7280">
        Average Dimension in Centimeters Across All 150 Samples
    </text>

    <!-- Legend -->
    <circle cx="210" cy="80" r="6" fill="#3b82f6"/>
    <text x="224" y="84" font-family="system-ui, sans-serif" font-size="11" fill="#374151">Iris Setosa</text>
    <circle cx="360" cy="80" r="6" fill="#10b981"/>
    <text x="374" y="84" font-family="system-ui, sans-serif" font-size="11" fill="#374151">Iris Versicolor</text>
    <circle cx="510" cy="80" r="6" fill="#8b5cf6"/>
    <text x="524" y="84" font-family="system-ui, sans-serif" font-size="11" fill="#374151">Iris Virginica</text>
"""
    feat_stats = payload["dataset_stats"]["feature_summary"]
    f_keys = list(feat_stats.keys())
    colors = ["#3b82f6", "#10b981", "#8b5cf6"]

    for i, fk in enumerate(f_keys):
        bx = 95 + i * 165
        svg_feat += f'<text x="{bx + 60}" y="325" text-anchor="middle" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="#4b5563">{fk.replace(" (cm)", "")}</text>\n'

        by_sp = feat_stats[fk]["by_species"]
        for j, spk in enumerate(species_names):
            m_val = by_sp[spk]["mean"]
            bar_h = (m_val / 8.0) * 190
            bar_y = 300 - bar_h
            bar_x = bx + j * 38
            svg_feat += f"""
            <rect x="{bar_x}" y="{bar_y:.1f}" width="28" height="{bar_h:.1f}" rx="4" fill="{colors[j]}"/>
            <text x="{bar_x + 14}" y="{bar_y - 5:.1f}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#374151">{m_val:.1f}</text>
            """

    svg_feat += "\n</svg>"

    with open(os.path.join(OUTPUTS_DIR, 'feature_distributions.svg'), 'w', encoding='utf-8') as f:
        f.write(svg_feat)
    with open(os.path.join(STATIC_IMG_DIR, 'feature_distributions.svg'), 'w', encoding='utf-8') as f:
        f.write(svg_feat)

    print("[✓] Visual charts generated in outputs/ and static/images/")


if __name__ == "__main__":
    train_and_evaluate()
