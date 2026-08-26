"""Leakage-aware benchmark for multiclass student outcome prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "Target"


@dataclass(frozen=True)
class ModelSpec:
    estimator: Pipeline
    parameters: dict[str, list[Any]]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the semicolon-delimited UCI dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    data = pd.read_csv(path, delimiter=";")
    data.columns = data.columns.str.strip()
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must include a {TARGET_COLUMN!r} column")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target labels must not be missing")
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("At least two target classes are required")
    return data


def prepare_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Encode labels and create one untouched stratified test split."""
    features = data.drop(columns=[TARGET_COLUMN]).copy()
    non_numeric = features.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric):
        features = pd.get_dummies(features, columns=list(non_numeric), dtype=float)
    encoder = LabelEncoder()
    target = encoder.fit_transform(data[TARGET_COLUMN].astype(str))
    split = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return (*split, encoder)


def build_model_specs(quick: bool = False) -> dict[str, ModelSpec]:
    """Return leakage-safe pipelines; SMOTE and scaling run inside each CV fold."""
    foldsafe_smote = ("smote", SMOTE(random_state=RANDOM_STATE))
    specs = {
        "random-forest": ModelSpec(
            Pipeline(
                [
                    foldsafe_smote,
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [150] if quick else [250, 500],
                "model__max_depth": [None] if quick else [12, None],
                "model__min_samples_leaf": [1] if quick else [1, 2],
            },
        ),
        "xgboost": ModelSpec(
            Pipeline(
                [
                    foldsafe_smote,
                    (
                        "model",
                        XGBClassifier(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100] if quick else [150, 300],
                "model__max_depth": [4] if quick else [3, 6],
                "model__learning_rate": [0.1] if quick else [0.05, 0.1],
            },
        ),
        "logistic": ModelSpec(
            Pipeline(
                [
                    foldsafe_smote,
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="saga",
                            max_iter=3000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [1.0] if quick else [0.1, 1.0, 10.0],
                "model__l1_ratio": [0.5] if quick else [0.1, 0.5, 0.9],
            },
        ),
        "mlp": ModelSpec(
            Pipeline(
                [
                    foldsafe_smote,
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            early_stopping=True,
                            max_iter=500,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [(80,)] if quick else [(80,), (100, 50)],
                "model__alpha": [0.001] if quick else [0.0001, 0.001, 0.01],
            },
        ),
    }
    return specs


def train_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    model_names: Sequence[str],
    quick: bool = False,
) -> dict[str, GridSearchCV]:
    specs = build_model_specs(quick)
    unknown = sorted(set(model_names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}")
    cross_validation = StratifiedKFold(
        n_splits=3 if quick else 5, shuffle=True, random_state=RANDOM_STATE
    )
    fitted = {}
    for name in model_names:
        spec = specs[name]
        search = GridSearchCV(
            spec.estimator,
            spec.parameters,
            scoring="f1_macro",
            cv=cross_validation,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        fitted[name] = search
    return fitted


def evaluate_models(
    fitted: dict[str, GridSearchCV],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    class_count: int,
) -> pd.DataFrame:
    rows = []
    for name, search in fitted.items():
        predictions = search.predict(x_test)
        probabilities = search.predict_proba(x_test)
        binarized = label_binarize(y_test, classes=np.arange(class_count))
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_macro": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "roc_auc_ovr_macro": roc_auc_score(
                    binarized, probabilities, multi_class="ovr", average="macro"
                ),
                "cv_f1_macro": search.best_score_,
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values(
        "f1_macro", ascending=False
    )


def save_results(
    output_dir: Path,
    results: pd.DataFrame,
    fitted: dict[str, GridSearchCV],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    encoder: LabelEncoder,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "metrics.csv")
    (output_dir / "metrics.json").write_text(
        json.dumps(results.reset_index().to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    best_name = results.index[0]
    best_search = fitted[best_name]
    joblib.dump(best_search.best_estimator_, output_dir / "best_model.joblib")
    matrix = confusion_matrix(y_test, best_search.predict(x_test))
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        ax=axis,
    )
    axis.set(title=f"Confusion Matrix - {best_name}", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=Path("data/student_dropout.csv"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(build_model_specs()),
        default=["random-forest", "xgboost", "logistic", "mlp"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one parameter setting and three-fold CV for a fast verification run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_dataset(args.data)
    x_train, x_test, y_train, y_test, encoder = prepare_split(data)
    fitted = train_models(x_train, y_train, args.models, args.quick)
    results = evaluate_models(fitted, x_test, y_test, len(encoder.classes_))
    save_results(args.output_dir, results, fitted, x_test, y_test, encoder)
    print(results.round(4).to_string())
    print(f"\nArtifacts written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
