# Application-Customer-Classification

This repository now contains a reproducible customer-churn classification submission built around the IBM Telco Customer Churn data.

## Deliverables
- Main report: `/home/runner/work/Application-Customer-Classification/Application-Customer-Classification/reports/knn_report.md`
- Separate comparison document: `/home/runner/work/Application-Customer-Classification/Application-Customer-Classification/reports/model_comparison.md`
- Reproducible analysis script: `/home/runner/work/Application-Customer-Classification/Application-Customer-Classification/analysis/generate_telco_report.py`
- Generated figures and metrics summary: `/home/runner/work/Application-Customer-Classification/Application-Customer-Classification/reports/figures/` and `/home/runner/work/Application-Customer-Classification/Application-Customer-Classification/reports/model_metrics.json`

## Reproducing the analysis
Install the dependencies and run:

```bash
pip install -r requirements.txt
python analysis/generate_telco_report.py
```

To use the originally attached Excel file instead of the default public CSV equivalent:

```bash
python analysis/generate_telco_report.py --input /absolute/path/to/CST-570-RS-WAFn-UseC-Telco-Customer-Churn.xlsx
```
