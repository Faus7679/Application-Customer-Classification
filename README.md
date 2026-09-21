# Application-Customer-Classification

This repository contains a reproducible customer churn classification analysis for the telecommunications assignment.

## Repository contents

- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` - repository-friendly CSV export of the provided telecom churn data
- `analysis.py` - prepares the data, tunes the kNN model, fits logistic regression, and saves figures/results
- `REPORT.md` - detailed assignment response with interpretation and references
- `COMPARISON.md` - separate detailed comparison of kNN and logistic regression
- `outputs/` - generated figures and metric summaries used in the report

## Run the analysis

```bash
python -m pip install -r requirements.txt
python analysis.py
```

The script writes refreshed figures and result tables to `outputs/`.
