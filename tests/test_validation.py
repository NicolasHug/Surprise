"""
Module for testing the validation module.
"""


import os

from surprise import Dataset, model_selection as ms, NormalPredictor, Reader


def test_cross_validate(toy_data):

    # First test with a specified CV iterator.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + "/custom_train", current_dir + "/custom_test")]

    reader = Reader(
        line_format="user item rating", sep=" ", skip_lines=3, rating_scale=(1, 5)
    )
    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    algo = NormalPredictor()
    pkf = ms.PredefinedKFold()
    ret = ms.cross_validate(algo, data, measures=["rmse", "mae"], cv=pkf, verbose=1)
    # Basically just test that keys (dont) exist as they should
    assert len(ret["test_rmse"]) == 1
    assert len(ret["test_mae"]) == 1
    assert len(ret["fit_time"]) == 1
    assert len(ret["test_time"]) == 1
    assert "test_fcp" not in ret
    assert "train_rmse" not in ret
    assert "train_mae" not in ret

    # Test that 5 fold CV is used when cv=None
    # Also check that train_* key exist when return_train_measures is True.
    ret = ms.cross_validate(
        algo,
        toy_data,
        measures=["rmse", "mae"],
        cv=None,
        return_train_measures=True,
        verbose=True,
    )
    assert len(ret["test_rmse"]) == 5
    assert len(ret["test_mae"]) == 5
    assert len(ret["fit_time"]) == 5
    assert len(ret["test_time"]) == 5
    assert len(ret["train_rmse"]) == 5
    assert len(ret["train_mae"]) == 5
