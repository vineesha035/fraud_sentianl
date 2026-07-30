# Financial Fraud Detection at Scale

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.4+-red.svg)](https://spark.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-orange.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Big Data Analytics & MLOps** — Distributed fraud detection pipeline using **Apache Spark**, **scikit-learn**, **XGBoost**, and **MLflow** on the IEEE-CIS Fraud Detection dataset.

<p align="center">
  <img src="assets/hero_fraud_detection.png" alt="Financial Fraud Detection at Scale" width="700"/>
</p>

---

## Overview

This project demonstrates end-to-end **data engineering** and **MLOps** for financial fraud detection:

- **Scale:** 590K+ training transactions, 400+ raw features, multi-table joins
- **Stack:** PySpark (Databricks/Spark), Delta Lake, Unity Catalog, MLflow, XGBoost
- **Outcome:** Champion model **AUC-ROC 0.946** (target > 0.90), with experiment tracking, model versioning, and a parameterized scoring pipeline

It is designed to showcase **Big Data** and **MLOps** competencies for data engineering and ML roles.

---

## Key Results

| Metric | Logistic Regression | Random Forest | **XGBoost (Champion)** |
|--------|----------------------|---------------|------------------------|
| **AUC-ROC** | 0.778 | 0.916 | **0.946** ✓ |
| AUC-PR | 0.178 | 0.582 | **0.675** |
| Precision @0.4 | 0.068 | 0.153 | **0.205** |
| Recall @0.4 | 0.804 | 0.836 | **0.867** |

- **Target:** AUC-ROC > 0.90 on held-out validation — **achieved** with XGBoost.
- **Best hyperparameters:** `learning_rate=0.1`, `max_depth=6`, `n_estimators=400` (3-fold GridSearchCV).

### Key visualizations

| Asset | Description |
|-------|-------------|
| [Hero image](assets/hero_fraud_detection.png) | Project banner — data + security theme |
| [Pipeline thumbnail](assets/pipeline_thumbnail.png) | End-to-end flow (ingest → Spark → ML → deploy) |
| `scripts/generate_assets.py` | Produces: class distribution, fraud by product/device, model comparison AUC, feature importance, precision-recall by threshold |

---

## Repository Structure

```
Financial-Fraud-Detection-at-Scale/
├── README.md                 # This file
├── requirements.txt           # Python dependencies
├── config/
│   └── paths.yaml             # Paths and experiment config (Databricks + local)
├── notebooks/
│   └── Financial_Fraud_Detection_at_Scale.ipynb   # Full pipeline notebook
├── scripts/
│   └── generate_assets.py    # Generate charts for README/docs (no data needed)
├── src/                       # Optional Python modules
├── assets/                    # Images and generated figures
│   ├── class_distribution.png
│   ├── fraud_by_product.png
│   ├── fraud_by_device.png
│   ├── model_comparison_auc.png
│   ├── feature_importance_correlation.png
│   ├── precision_recall_by_threshold.png
│   └── hero_fraud_detection.png   # AI-generated hero image
├── docs/
│   ├── ARCHITECTURE.md        # Pipeline and tech stack
│   └── DATA.md                # Dataset and schema
└── data/                      # Raw/processed data (gitignored; add via Kaggle)
    ├── raw/
    └── processed/
```

---

## Pipeline Overview

| Phase | Description |
|-------|-------------|
| **1. Problem & data** | IEEE-CIS dataset selection, business KPIs (AUC-ROC > 0.90), Unity Catalog paths |
| **2. Ingestion & preprocessing** | Spark CSV read, Hive views, LEFT JOIN transaction + identity, dedup, null handling (drop >60% null, median/mode impute), label encoding, Parquet/Delta write |
| **3. EDA** | Class imbalance, TransactionAmt by label, fraud rate by ProductCD/DeviceType/email/hour, correlation heatmaps, missing-data heatmap, 10 business insights |
| **4. Modeling** | Train/val split (80/20), StandardScaler, Logistic Regression, Random Forest, XGBoost (GridSearchCV), MLflow logging (params, metrics, models) |
| **5. MLOps** | Champion selection (XGBoost), model + scaler + metadata to Unity Catalog Volume, experiment summary CSV, feature importance & evaluation plots, widget-based scoring pipeline |

---

## Dataset

- **Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle, IEEE Computational Intelligence Society)
- **Tables:** `train_transaction`, `train_identity`, `test_transaction`, `test_identity` (join key: `TransactionID`)
- **Target:** `isFraud` (binary); ~96.5% legitimate, ~3.5% fraud (highly imbalanced)
- **Size:** 590,540 training rows; 394 transaction + 41 identity columns before join

See [docs/DATA.md](docs/DATA.md) for schema and usage notes.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/Financial-Fraud-Detection-at-Scale.git
cd Financial-Fraud-Detection-at-Scale
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

- Download the dataset from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data).
- Place `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv` in `data/raw/` (or upload to your Databricks Unity Catalog Volume as in the notebook).

### 3. Run the notebook

- **Databricks (recommended):** Import `notebooks/Financial_Fraud_Detection_at_Scale.ipynb` into a Databricks workspace, attach to a cluster with Spark 3.x+ and the same dependencies, and run. Configure Unity Catalog Volume paths in the notebook to match your environment.
- **Local:** Use a Spark 3.x installation and point the notebook to `data/raw/` and local output paths (see `config/paths.yaml`).

### 4. Regenerate assets (optional)

From the repo root, with dependencies installed (`pip install -r requirements.txt` or at least `numpy`, `matplotlib`, `seaborn`):

```bash
python scripts/generate_assets.py
```

This writes the charts under `assets/` using summary statistics only (no raw data). The repo may already include pre-generated assets.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Compute / orchestration | Apache Spark (Databricks), Python 3.10+ |
| Storage / catalog | Unity Catalog Volumes, Delta Lake / Parquet |
| ML framework | scikit-learn, XGBoost |
| Experiment tracking | MLflow (Databricks + file-backed) |
| Visualization | matplotlib, seaborn |

---

## Business Context

- **Problem:** Payment card fraud losses exceeded **$33B in 2023**; rule-based systems do not scale to modern volume or evolving patterns.
- **Goal:** ML-based risk scoring with **AUC-ROC > 0.90**, high recall at acceptable precision to minimize fraud loss while limiting false positives (~$8–$12 per false positive in servicing cost).
- **Deliverables:** Champion model (XGBoost), MLflow runs, model + scaler + metadata in catalog, and a parameterized scoring pipeline (threshold, input table).

---

## Contributors

**Group 5 — Big Data Analytics & MLOps**

- Satish  
- Debadri Sanyal  
- Sara Tariq  
- Prerna Jha  

---

## License

This project is available under the [MIT License](LICENSE).

---

## References

- [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
