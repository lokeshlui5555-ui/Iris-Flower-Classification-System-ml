# IrisAI – Intelligent Iris Flower Classification System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://palletsprojects.com/p/flask/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-96.7%25-brightgreen.svg)]()

**IrisAI** is a machine learning web application that classifies Iris flower species from four morphological dimensions. Designed with a modern, responsive purple-and-lavender SaaS dashboard aesthetic, the platform combines supervised learning algorithms with real-time inference and interactive Chart.js visualizations.

---

## 🌸 Key Features

1. **Machine Learning Classifiers**:
   - Compares **4 classic algorithms**: Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Logistic Regression.
   - Evaluated using 80/20 stratified train-test splits and **5-Fold Stratified Cross-Validation**.
   - Champion model (**Random Forest**) delivers **96.7% test accuracy** and **96.0% mean CV score**.
   - Feature scaling managed via `StandardScaler` to prevent leakage.

2. **Interactive Prediction Interface**:
   - Synchronized numeric inputs and range sliders for fine-grained measurement adjustments.
   - One-click sample presets for *Setosa*, *Versicolor*, and *Virginica*.
   - Live probability distribution bar chart and confidence meter.
   - Rich botanical profiles with custom vector illustrations.

3. **Analytics Dashboard**:
   - Comparative algorithm performance benchmarks.
   - 3×3 multi-class confusion matrix heatmap.
   - Biometric feature distributions by species.
   - Balanced species breakdown charts.

4. **REST API & CLI**:
   - `/api/predict` endpoint accepting JSON payloads with validation.
   - Standalone CLI inference module `predict.py` for headless scripting.

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons
- **Visualizations**: Chart.js 4.4, Custom SVG Charts, Matplotlib & Seaborn
- **Backend**: Python 3.9+, Flask 3.1
- **Machine Learning**: Scikit-learn, NumPy, Pandas, Joblib

---

## 📁 Project Directory Structure

```text
IrisAI/
├── app.py                     # Flask web server and routing controller
├── train_model.py             # ML pipeline: data ingestion, training, evaluation, benchmark
├── predict.py                 # Standalone prediction module and CLI tool
├── requirements.txt           # Python package dependencies
├── README.md                  # System documentation and execution guide
├── dataset/
│   └── iris.csv               # Exported Fisher's 150-sample Iris dataset
├── models/
│   ├── best_model.joblib      # Champion trained classifier artifact
│   ├── scaler.joblib          # Fitted StandardScaler artifact
│   ├── metrics.json           # Benchmark statistics and CV evaluation results
│   ├── decision_tree.joblib   # Saved individual model
│   ├── random_forest.joblib   # Saved individual model
│   ├── k-nearest_neighbors.joblib
│   └── logistic_regression.joblib
├── outputs/
│   ├── model_comparison.svg   # Benchmark accuracy & CV comparison chart
│   ├── confusion_matrix.svg   # Multi-class test confusion matrix
│   └── feature_distributions.svg # Morphological feature dimension breakdown
├── static/
│   ├── css/
│   │   └── style.css          # Purple/Lavender modern SaaS design system
│   ├── js/
│   │   └── script.js          # Interactive forms, slider sync, AJAX, Chart.js
│   └── images/
│       ├── logo.svg           # IrisAI brand vector logo
│       ├── setosa.svg         # Iris Setosa botanical illustration
│       ├── versicolor.svg     # Iris Versicolor botanical illustration
│       ├── virginica.svg      # Iris Virginica botanical illustration
│       ├── hero_iris.svg      # Hero illustration with AI tech rings
│       ├── confusion_matrix.svg
│       └── model_comparison.svg
└── templates/
    ├── index.html             # Landing page with hero & statistics
    ├── prediction.html        # Interactive prediction interface
    ├── analytics.html         # Model evaluation dashboard & charts
    └── about.html             # Botanical history, math methodology & architecture
```

---

## 🚀 Getting Started with VS Code

### 1. Prerequisites
Ensure Python 3.9+ is installed on your computer. Verify in your terminal:
```bash
python --version
```

### 2. Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Go to **File** → **Open Folder...** (or press `Ctrl + K, Ctrl + O`).
3. Select the `IrisAI` directory.

### 3. Set Up a Virtual Environment (Recommended)
Open the integrated terminal in VS Code (`Ctrl + ~`) and execute:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Option A: Run the Flask Web Application
To start the web server:
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

### Option B: Retrain & Benchmark Models
To execute the complete machine learning training pipeline:
```bash
python train_model.py
```
This updates the models in `models/`, writes `models/metrics.json`, and updates charts in `outputs/` and `static/images/`.

### Option C: Run CLI Predictions
Test predictions directly from your terminal:
```bash
# Predict Setosa
python predict.py -sl 5.1 -sw 3.5 -pl 1.4 -pw 0.2

# Predict Versicolor
python predict.py -sl 6.0 -sw 2.7 -pl 4.0 -pw 1.3

# Predict Virginica
python predict.py -sl 6.5 -sw 3.0 -pl 5.5 -pw 2.0
```

---

## 🧪 Benchmark Comparison

| Algorithm | Train Acc | Test Acc | 5-Fold CV Mean | Precision | Recall | F1-Score | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Classifier** | **100.0%** | **96.7%** | **96.0% (±3.89%)** | **0.9667** | **0.9667** | **0.9666** | 🏆 **Champion** |
| **K-Nearest Neighbors (KNN)** | 97.5% | 93.3% | 97.3% (±2.49%) | 0.9333 | 0.9333 | 0.9327 | Candidate |
| **Decision Tree Classifier** | 99.2% | 93.3% | 95.3% (±3.40%) | 0.9333 | 0.9333 | 0.9333 | Candidate |
| **Logistic Regression** | 95.8% | 93.3% | 95.3% (±4.52%) | 0.9333 | 0.9333 | 0.9333 | Candidate |

---

## 📊 Sample Prediction Test Cases

Use these representative flower measurements to test the system:

| Species | Sepal Length | Sepal Width | Petal Length | Petal Width | Expected Species |
|:---|:---:|:---:|:---:|:---:|:---|
| **Sample A** | `5.1 cm` | `3.5 cm` | `1.4 cm` | `0.2 cm` | **Iris Setosa** |
| **Sample B** | `4.9 cm` | `3.0 cm` | `1.4 cm` | `0.2 cm` | **Iris Setosa** |
| **Sample C** | `6.0 cm` | `2.7 cm` | `4.0 cm` | `1.3 cm` | **Iris Versicolor** |
| **Sample D** | `5.7 cm` | `2.8 cm` | `4.5 cm` | `1.3 cm` | **Iris Versicolor** |
| **Sample E** | `6.5 cm` | `3.0 cm` | `5.5 cm` | `2.0 cm` | **Iris Virginica** |
| **Sample F** | `7.7 cm` | `3.8 cm` | `6.7 cm` | `2.2 cm` | **Iris Virginica** |

---

## 🌐 API Reference

### `POST /api/predict`
Accepts 4 biometric measurements in a JSON body.

**Request:**
```json
{
  "sepal_length": 6.5,
  "sepal_width": 3.0,
  "petal_length": 5.5,
  "petal_width": 2.0
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "predicted_species": "Iris Virginica",
  "species_id": 2,
  "confidence": 99.72,
  "model_name": "Random Forest",
  "probabilities": {
    "Iris Setosa": 0.0,
    "Iris Versicolor": 0.28,
    "Iris Virginica": 99.72
  },
  "features": {
    "sepal_length": 6.5,
    "sepal_width": 3.0,
    "petal_length": 5.5,
    "petal_width": 2.0
  },
  "metadata": {
    "scientific_name": "Iris virginica L.",
    "common_name": "Virginia Iris / Great Blue Flag",
    "description": "Iris Virginica possesses the largest and most robust petals in the Iris family...",
    "distinct_traits": "Petal length > 4.5 cm and petal width > 1.4 cm...",
    "typical_habitat": "Lowland swamps, river floodplains, and coastal freshwater retention basins..."
  }
}
```

---

## 📄 License
This project is open-source under the MIT License.
