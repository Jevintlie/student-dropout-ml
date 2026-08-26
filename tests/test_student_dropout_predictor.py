from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_dropout_predictor import (
    build_model_specs,
    load_dataset,
    prepare_split,
)


DATASET = Path(__file__).parents[1] / "data" / "student_dropout.csv"


def test_uci_dataset_shape_and_target_classes():
    data = load_dataset(DATASET)
    assert data.shape == (4424, 37)
    assert set(data["Target"]) == {"Dropout", "Enrolled", "Graduate"}
    assert all(column == column.strip() for column in data.columns)


def test_split_is_reproducible_and_stratified():
    data = load_dataset(DATASET)
    first = prepare_split(data)
    second = prepare_split(data)
    x_train, x_test, y_train, y_test, encoder = first

    assert x_train.index.equals(second[0].index)
    assert x_test.index.equals(second[1].index)
    assert encoder.classes_.tolist() == ["Dropout", "Enrolled", "Graduate"]
    assert len(x_train) + len(x_test) == len(data)
    overall = np.bincount(np.concatenate([y_train, y_test])) / len(data)
    test_ratio = np.bincount(y_test) / len(y_test)
    assert np.allclose(overall, test_ratio, atol=0.02)


def test_training_pipelines_keep_smote_inside_cross_validation():
    for spec in build_model_specs(quick=True).values():
        assert list(spec.estimator.named_steps)[0] == "smote"
        assert all(len(values) == 1 for values in spec.parameters.values())


def test_loader_rejects_missing_target(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError, match="Target"):
        load_dataset(path)
