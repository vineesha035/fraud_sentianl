# Dataset — IEEE-CIS Fraud Detection

## Source

- **Competition:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle, IEEE Computational Intelligence Society)
- **Files:** `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`
- **Join key:** `TransactionID` (present in both transaction and identity tables)

## Scale (from project run)

| Table | Rows | Columns |
|-------|------|---------|
| train_transaction | 590,540 | 394 |
| train_identity | 144,233 | 41 |
| test_transaction | 506,691 | 393 |
| test_identity | 141,907 | 41 |
| train_joined (after LEFT JOIN) | 590,540 | 434 |

## Target

- **Variable:** `isFraud` (integer: 0 = legitimate, 1 = fraudulent)
- **Distribution:** ~96.5% legitimate, ~3.5% fraud (highly imbalanced — AUC-ROC and AUC-PR are primary metrics, not accuracy)

## Main Feature Groups

| Group | Examples | Notes |
|-------|----------|--------|
| Transaction | TransactionID, TransactionDT, TransactionAmt, ProductCD | Core transaction fields |
| Card | card1–card6 | Card attributes (some encoded) |
| Address | addr1, addr2 | Billing/shipping |
| C-series | C1–C14 | Aggregate counts (strong correlation with isFraud) |
| D-series | D1–D15 | Time/delta features |
| V-series | V1–V339 | Anonymized Vesta features (many with high null %) |
| Identity | id_01–id_38, DeviceType, DeviceInfo, P_emaildomain, etc. | Identity/device (identity table) |

## Preprocessing (project choices)

- **Nulls:** Columns with >60% missing were dropped; remaining numeric nulls median-imputed, categorical nulls mode-imputed (train stats applied to test).
- **Encoding:** ProductCD, card4, card6, DeviceType, P_emaildomain — label-encoded (fit on train+test combined uniques).
- **Feature set for modeling:** 43 numeric features (including encoded indices); TransactionID, TransactionDT, and raw string columns dropped for training.

## Usage

1. Download from Kaggle (requires account and competition acceptance).
2. Place CSVs in `data/raw/` for local runs, or upload to your Databricks Unity Catalog Volume as in the notebook.
3. Notebook expects either Unity Catalog table names (if tables are registered from the Volume) or CSV paths — adjust the first ingestion cell accordingly.

## License / attribution

Dataset is provided by IEEE-CIS for the Kaggle competition; see the competition page for terms and citation.
