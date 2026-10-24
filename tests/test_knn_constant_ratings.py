import numpy as np
import pandas as pd

import pytest

from surprise import Dataset, KNNWithZScore, Reader


@pytest.mark.parametrize("rating", [-2, 0, 3])
@pytest.mark.parametrize("user_based", [True, False])
@pytest.mark.parametrize("clip", [True, False])
def test_zscore_constant_ratings(rating, user_based, clip):
    # The requested pair is missing, but both ids and a neighbor are known.
    frame = pd.DataFrame(
        [("u1", "i1", rating), ("u2", "i1", rating), ("u2", "i2", rating)]
    )
    data = Dataset.load_from_df(frame, Reader(rating_scale=(-5, 5)))
    algo = KNNWithZScore(sim_options={"user_based": user_based}, verbose=False)
    algo.fit(data.build_full_trainset())

    prediction = algo.predict("u1", "i2", clip=clip)

    assert np.isfinite(prediction.est)
    assert prediction.est == rating
    assert not prediction.details["was_impossible"]
    assert prediction.details["actual_k"] == 1


@pytest.mark.parametrize("rating", [0.1, 0.2, 0.3, -0.1])
@pytest.mark.parametrize("extra_ratings", [1, 5])
@pytest.mark.parametrize("user_based", [True, False])
@pytest.mark.parametrize("clip", [True, False])
def test_zscore_constant_decimals(rating, extra_ratings, user_based, clip):
    # Uneven counts can give identical decimals different rounded means/sigmas.
    rows = [
        ("u1", "i1", rating),
        ("u1", "i2", rating),
        ("u2", "i1", rating),
        ("u2", "i2", rating),
        ("u2", "i3", rating),
    ] + [("u3", f"i{i}", rating) for i in range(4, 4 + extra_ratings)]
    uid, iid = "u1", "i3"
    if not user_based:
        rows = [(i, u, r) for u, i, r in rows]
        uid, iid = iid, uid
    data = Dataset.load_from_df(pd.DataFrame(rows), Reader(rating_scale=(-1, 1)))
    algo = KNNWithZScore(sim_options={"user_based": user_based}, verbose=False)
    algo.fit(data.build_full_trainset())

    prediction = algo.predict(uid, iid, clip=clip)

    assert prediction.est == rating
    assert prediction.details == {"actual_k": 1, "was_impossible": False}


@pytest.mark.parametrize("user_based", [True, False])
def test_zscore_small_nonzero_variance(user_based):
    # Real variation must retain its scale, even far below a typical epsilon.
    rows = [
        ("u1", "i1", 0.0),
        ("u1", "i2", 2e-12),
        ("u2", "i1", 0.0),
        ("u2", "i2", 2e-12),
        ("u2", "i3", 4e-12),
    ]
    uid, iid = "u1", "i3"
    if not user_based:
        rows = [(i, u, r) for u, i, r in rows]
        uid, iid = iid, uid
    data = Dataset.load_from_df(pd.DataFrame(rows), Reader(rating_scale=(0, 1)))
    algo = KNNWithZScore(sim_options={"user_based": user_based}, verbose=False)
    algo.fit(data.build_full_trainset())

    prediction = algo.predict(uid, iid, clip=False)

    # Target mean/sigma are 1e-12; the neighbor's z-score is sqrt(3 / 2).
    expected = (1 + np.sqrt(3 / 2)) * 1e-12
    assert prediction.est == pytest.approx(expected, rel=1e-14, abs=0)
    assert prediction.details == {"actual_k": 1, "was_impossible": False}
