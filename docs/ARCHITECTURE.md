# Architecture — Financial Fraud Detection at Scale

## High-Level Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw CSVs       │────▶│  Spark Ingest     │────▶│  Join + Clean    │
│  (Unity Vol /   │     │  Hive views       │     │  Impute + Encode │
│   data/raw)     │     │  Parquet/Delta    │     │  Feature tables  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌──────────────────┐             │
│  MLOps          │◀────│  Train / Val     │◀────────────┘
│  MLflow, model  │     │  LR, RF, XGBoost │
│  registry,      │     │  GridSearchCV     │
│  scoring        │     │  Metrics (AUC)    │
└─────────────────┘     └──────────────────┘
```

## Components

| Layer | Role |
|-------|------|
| **Storage** | Unity Catalog Volumes (Databricks) or local `data/` — raw CSVs, processed Parquet/Delta, model artifacts |
| **Compute** | Spark driver/executors for ingestion, joins, aggregation; single-node scikit-learn/XGBoost for training (on collected sample or downsampled data) |
| **Catalog** | Hive metastore / Unity Catalog — table registration for `raw_*`, `train_joined_view`, `fraud_features_train/test` |
| **Experiment tracking** | MLflow — params, metrics, and model artifacts per run; Databricks Experiments or file-backed `mlruns/` |
| **Serving** | Batch scoring via notebook widget (threshold, input table) loading champion model + scaler from Volume |

## Data Flow

1. **Ingest:** Read 4 CSVs from Volume or `data/raw/` into Spark DataFrames; register as Hive temp views.
2. **Join:** `train_transaction` LEFT JOIN `train_identity` on `TransactionID` (same for test).
3. **Clean:** Drop duplicates; drop columns with >60% nulls; median-impute numeric, mode-impute categorical nulls.
4. **Encode:** Label-encode categoricals (ProductCD, card4, card6, DeviceType, P_emaildomain); write processed tables.
5. **Feature table:** Drop IDs/timestamps and raw strings; persist `fraud_features_train` / `fraud_features_test` (Delta or Parquet).
6. **Train:** Load feature table to Pandas; train/val split; StandardScaler; train LR, RF, XGBoost; log to MLflow; select champion (best AUC-ROC).
7. **MLOps:** Save champion model, scaler, and metadata JSON to Volume; generate evaluation plots; optional widget-based batch scoring.

## Technologies

- **Apache Spark 3.4+ / 4.x** — distributed read, join, aggregation, null analysis, Parquet/Delta write
- **Delta Lake / Parquet** — columnar storage for processed and feature tables
- **scikit-learn** — StandardScaler, train_test_split, LogisticRegression, RandomForestClassifier, GridSearchCV
- **XGBoost** — GradientBoostingClassifier with `tree_method="hist"`
- **MLflow** — experiment and run tracking; model logging (sklearn flavor)
- **Databricks** — optional: Unity Catalog, serverless compute, Experiments UI

## Security & Operations

- Data and models live in Unity Catalog Volumes (or local paths) with access controlled by workspace/catalog permissions.
- No secrets in code; paths and experiment name come from config (e.g. `config/paths.yaml`).
- Scoring pipeline is parameterized (threshold, input table) for reuse across environments.
