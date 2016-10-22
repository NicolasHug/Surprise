"""
Module for testing the Dataset class
"""

import os
import pytest

from pyrec import Dataset
from pyrec import Reader


reader = Reader(line_format='user item rating', sep=' ', skip_lines=3)
file_path = os.path.dirname(os.path.realpath(__file__)) + '/custom_dataset'

def test_split():
    """Test the split method."""

    data = Dataset.load_from_file(file_path=file_path, reader=reader)

    # Test n_folds parameter
    data.split(5)
    assert len(list(data.folds)) == 5

    with pytest.raises(ValueError):
        data.split(10)
        for fold in data.folds:
            pass

    with pytest.raises(ValueError):
        data.split(1)
        for fold in data.folds:
            pass

    # Test the shuffle parameter
    data.split(n_folds=3, shuffle=False)
    testsets_a = [testset for (_, testset) in data.folds]
    data.split(n_folds=3, shuffle=False)
    testsets_b = [testset for (_, testset) in data.folds]
    assert testsets_a == testsets_b

    data.split(n_folds=3, shuffle=True)
    testsets_b = [testset for (_, testset) in data.folds]
    assert testsets_a != testsets_b
    # Note : there's a chance that the above test fails, just by lack of luck.
    # This is probably not a good thing.

    # Ensure that folds are the same if split is not called again
    testsets_a = [testset for (_, testset) in data.folds]
    testsets_b = [testset for (_, testset) in data.folds]
    assert testsets_a == testsets_b
