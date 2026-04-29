# Customer Churn Risk Model - Beta Bank

Machine learning system for identifying high-risk banking customers and supporting targeted retention decisions.

## Project Access

- English documentation: [README_EN.md](README_EN.md)
- Documentación en español: [README_ES.md](README_ES.md)
- Notebook, English: [notebooks/project_EN.ipynb](notebooks/project_EN.ipynb)
- Notebook, Español: [notebooks/project_ES.ipynb](notebooks/project_ES.ipynb)

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/project_EN.ipynb
```

The notebook expects the dataset at:

```text
data/Churn.csv
```

## What This Project Demonstrates

- Customer churn risk modeling
- Class imbalance handling
- Threshold optimization beyond the default 0.5 cutoff
- F1-score and AUC-ROC evaluation
- Reproducible preprocessing and modeling pipelines
- Business-aligned customer risk prioritization

## Key Insight

The model should be used as a prioritization layer for retention teams, not as an isolated automated decision system. It helps identify which customers deserve attention first when intervention resources are limited.
