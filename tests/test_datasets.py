"""Unit tests for label and split construction (no downloads)."""

import pickle

import numpy as np
import pytest

from hfd.datasets import _load_blogcatalog_split, even_quantile_labels


def test_even_quantile_labels_balanced():
    vals = np.arange(1000, dtype=np.float64)
    labels = even_quantile_labels(vals, 5)
    counts = np.bincount(labels)
    assert len(counts) == 5
    assert counts.tolist() == [200] * 5
    # Monotone in the underlying value.
    assert (np.diff(labels) >= 0).all()


def test_even_quantile_labels_all_assigned():
    rng = np.random.RandomState(0)
    vals = rng.randint(1990, 2021, size=5000).astype(np.float64)
    labels = even_quantile_labels(vals, 5)
    assert (labels >= 0).all() and (labels <= 4).all()


def _write_split(path, parts) -> None:
    with path.open("wb") as handle:
        pickle.dump(parts, handle)


def test_blogcatalog_public_split_is_exhaustive_and_has_fixed_sizes(tmp_path) -> None:
    path = tmp_path / "split.pkl"
    expected = (
        np.arange(0, 519),
        np.arange(519, 1558),
        np.arange(1558, 5196),
    )
    _write_split(path, expected)

    observed = _load_blogcatalog_split(path, 5196)

    assert tuple(map(len, observed)) == (519, 1039, 3638)
    assert np.array_equal(np.concatenate(observed), np.arange(5196))


@pytest.mark.parametrize(
    "parts",
    [
        (np.array([0, 1]), np.array([2]), np.array([2, 3])),
        (np.array([0, 1]), np.array([2]), np.array([4, 5])),
        (np.array([0]), np.array([1]), np.array([2]), np.array([3])),
    ],
)
def test_blogcatalog_public_split_rejects_invalid_partitions(tmp_path, parts) -> None:
    path = tmp_path / "split.pkl"
    _write_split(path, parts)

    with pytest.raises(ValueError):
        _load_blogcatalog_split(path, 4)
