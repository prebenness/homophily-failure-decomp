import numpy as np
import torch

from hfd.dataset_readiness import inspect_dataset_readiness
from hfd.datasets import DatasetBundle, Split


def test_readiness_accepts_a_complete_thirty_run_panel() -> None:
    # Every node has local homophily 1/3. The empty base bins merge under the
    # registered rule, and every split partition still has all three classes.
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    edges = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6, 7, 8],
            [2, 3, 4, 5, 0, 1, 7, 8, 6, 10, 11, 9, 1, 3, 5, 9, 10, 11],
        ],
        dtype=np.int64,
    )
    split = Split(
        split_id="toy-0",
        train_idx=np.array([0, 2, 4, 6, 7, 8]),
        val_idx=np.array([1, 3, 5]),
        test_idx=np.array([9, 10, 11]),
    )
    bundle = DatasetBundle(
        name="toy",
        x=torch.ones((12, 2)),
        y=y,
        edge_index=edges,
        metric="accuracy",
        splits=[split] * 10,
        training_seeds=[1, 2, 3],
        split_panel="toy",
    )

    report = inspect_dataset_readiness(bundle)

    assert report["n_runs_per_model_preparation"] == 30
    assert report["graph"]["num_classes"] == 3
    assert report["local_homophily"]["direction"] == "heterophilic"
    assert report["ready"] is True
    assert report["errors"] == []
    assert len(report["local_homophily"]["merged_bins"]) == 1
    assert len(report["splits"]) == 10


def test_readiness_rejects_out_of_range_split_indices() -> None:
    bundle = DatasetBundle(
        name="invalid",
        x=torch.ones((3, 2)),
        y=np.array([0, 1, 2]),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        metric="accuracy",
        splits=[
            Split(
                split_id="bad",
                train_idx=np.array([0, 3]),
                val_idx=np.array([1]),
                test_idx=np.array([2]),
            )
        ],
        training_seeds=list(range(1, 31)),
        split_panel="invalid",
    )

    with np.testing.assert_raises_regex(ValueError, "out-of-range"):
        inspect_dataset_readiness(bundle)
