# Student Dropout and Academic Success Prediction

This project benchmarks four supervised-learning approaches for predicting whether a higher-education student will drop out, remain enrolled, or graduate. It was created by Jevint Felixciano for SWA2124 Social and Web Analytics and has been rebuilt as a deterministic, leakage-aware command-line experiment.

## Why this version is stronger

The coursework prototype applied SMOTE and feature scaling before cross-validation, which could leak information between validation folds. The portfolio version places SMOTE and scaling inside each imbalanced-learn pipeline so every fold learns preprocessing only from its own training partition. A separate stratified 20% test set remains untouched until final evaluation.

## Models

- Random Forest
- XGBoost
- Elastic-net logistic regression
- Multilayer perceptron

Model selection uses stratified cross-validation with macro F1, which gives each outcome class equal importance despite class imbalance. Final reporting includes accuracy, macro precision/recall/F1, one-vs-rest macro ROC AUC, and cross-validation macro F1.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run

Fast verification:

```bash
python student_dropout_predictor.py --quick --models logistic random-forest
```

Full benchmark:

```bash
python student_dropout_predictor.py
```

Results are written to `artifacts/` as CSV, JSON, a serialized best pipeline, and a confusion-matrix image.

## Verified quick-run result

The checked-in artifacts come from the fast verification command above with a fixed random seed. On the untouched 20% test partition, Random Forest was the stronger of the two quick-run models:

| Model | Accuracy | Macro F1 | Macro one-vs-rest ROC AUC |
| --- | ---: | ---: | ---: |
| Random Forest | 0.754 | 0.703 | 0.881 |
| Logistic regression | 0.732 | 0.692 | 0.867 |

These figures verify that the pipeline runs correctly; they are not presented as exhaustive tuning or evidence of deployment readiness.

## Test

```bash
pytest
```

## Dataset

The included 4,424-row dataset is provided by the UCI Machine Learning Repository under CC BY 4.0. It contains 36 academic, demographic, and socioeconomic features. Full attribution, DOI, license, and file hash are documented in [`data/README.md`](data/README.md).

## Responsible-use limitations

This is an educational benchmark, not a production decision system. Historical labels can encode institutional and socioeconomic bias; demographic features may create disparate impact; and performance on one Portuguese institution does not establish generalization elsewhere. Any real intervention should be supportive, human-reviewed, transparent to students, regularly audited for subgroup performance, and never used as the sole basis for punitive decisions.
