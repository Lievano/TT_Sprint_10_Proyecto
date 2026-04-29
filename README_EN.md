# Customer Churn Risk Model — Beta Bank

## Executive Summary

This project builds a machine learning pipeline to identify banking customers with high probability of churn. It is framed as a customer risk prioritization system: the goal is not only to classify customers, but to help a retention team decide where to focus intervention resources first.

The workflow compares multiple modeling strategies, handles class imbalance, optimizes the decision threshold for F1-score, and evaluates performance on a held-out test set.

## Business Problem

Customer churn increases acquisition costs, reduces customer lifetime value, and weakens revenue stability. A retention team needs a reliable way to identify customers who are likely to leave before the relationship is lost.

The model supports this by ranking customers according to churn risk.

## Technical Objective

Predict the target variable `Exited`:

- `0` = customer stayed
- `1` = customer churned

The primary metric is F1-score because the dataset is imbalanced and both false positives and false negatives matter. AUC-ROC is also reported to evaluate ranking quality.

## Dataset

The dataset contains customer-level banking information:

| Feature | Meaning |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Country of residence |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is active |
| `EstimatedSalary` | Estimated salary |
| `Exited` | Target variable |

Identifier fields such as `RowNumber`, `CustomerId`, and `Surname` are removed because they do not provide meaningful predictive signal.

## Methodology

The project follows a reproducible machine learning workflow:

1. Load and inspect the dataset.
2. Remove non-predictive identifiers.
3. Separate numerical and categorical variables.
4. Apply preprocessing with scikit-learn pipelines.
5. Split the data into train, validation, and test sets.
6. Compare several imbalance-handling strategies.
7. Tune selected models.
8. Optimize the classification threshold.
9. Evaluate final performance on the test set.

## Modeling

The project evaluates Logistic Regression and Random Forest models with three imbalance strategies:

| Strategy | Purpose |
|---|---|
| `class_weight` | Penalizes errors on the minority class more heavily |
| `SMOTE-NC` | Creates synthetic minority samples for mixed numerical and categorical data |
| `RandomUnderSampler` | Reduces majority-class dominance |

Random Forest becomes the strongest model family because it captures non-linear interactions between customer attributes.

## Validation

The data is split into train, validation, and test sets using stratification to preserve the churn ratio.

The selected model is chosen based on validation F1-score. Instead of relying on the default `0.5` threshold, the project uses the precision-recall curve to identify the threshold that maximizes F1.

## Results

| Metric | Approximate Result |
|---|---|
| F1-score on test | ~0.63-0.67 |
| AUC-ROC | ~0.85 |
| Minimum target F1 | 0.59 |
| Strongest model family | Random Forest with imbalance handling |

The final model exceeds the required F1 threshold.

## Insights

The strongest churn signals tend to include:

- customer age
- activity status
- number of products
- geography, especially the Germany segment
- balance-related behavior

These patterns suggest that churn risk is both behavioral and product-related, not merely demographic.

## Impact

This model can support:

- targeted retention campaigns
- customer risk tiers
- proactive outreach workflows
- reduction of wasted retention budget
- periodic churn-risk monitoring

The output should be treated as a decision-support layer, not as an automatic final decision.

## Repository Structure

```text
customer-churn-risk-model/
├── data/
│   └── Churn.csv
├── notebooks/
│   ├── project_EN.ipynb
│   └── project_ES.ipynb
├── src/
│   └── pipeline.py
├── reports/
│   └── figures/
├── README.md
├── README_EN.md
├── README_ES.md
├── docs/
│   ├── INTERVIEW_EN.md
│   ├── INTERVIEW_ES.md
│   ├── results_summary.md
│   ├── notes.md
│   └── figures/
├── requirements.txt
└── .gitignore
```

Important: `docs/` exists locally for interview preparation and deeper notes, but it is ignored by git.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the English notebook:

```bash
jupyter notebook notebooks/project_EN.ipynb
```

Or open the Spanish notebook:

```bash
jupyter notebook notebooks/project_ES.ipynb
```

## Next Steps

1. Add behavioral features, such as `Balance / EstimatedSalary`.
2. Test gradient boosting models.
3. Calibrate probabilities before using scores operationally.
4. Add drift monitoring for churn rate and feature distributions.
5. Package the final pipeline for scheduled scoring.
