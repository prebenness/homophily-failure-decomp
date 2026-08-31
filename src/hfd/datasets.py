"""Dataset loaders and split panels.

Loads the five core datasets with the split panels fixed in the proposal's
Experimental Contract:

- PubMed: PyG Planetoid with split="geom-gcn", 10 public splits, seeds 1-3.
- ogbn-arxiv: official OGB time split, seeds 1-30.
- Roman-empire / Amazon-ratings: the 10 released Platonov-suite 50/25/25
  splits, seeds 1-3.
- arxiv-year: the ogbn-arxiv graph with 5-class even-quantile year labels
  (Lim et al.), the 5 fixed 50/25/25 splits released in the
  Non-Homophily-Large-Scale repository, seeds 1-6.

The post-review extension registers two further multiclass heterophilic data
sets in a separate experiment panel:

- Actor (also called Film): PyG's Geom-GCN data and 10 released splits
  (48%/32%/20% train/validation/test), seeds 1-3.
- BlogCatalog: PyG's attributed-graph data and the fixed 10/20/70 split used
  by GAug and OpenGSL, seeds 1-30.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ARXIV_YEAR_NUM_CLASSES = 5
ARXIV_YEAR_SPLITS_URL = (
    "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
    "arxiv-year-splits.npy"
)
ARXIV_YEAR_SPLITS_SHA256 = (
    "a2d4c8b2ed621270642dcf4886c3f6e425da664b2d31c7a28fb11a6da763077a"
)
BLOGCATALOG_SPLIT_URL = (
    "https://raw.githubusercontent.com/zhao-tong/GAug/"
    "e7d7deab818b822942ba0d20a508ae2cf709a39c/data/graphs/"
    "blogcatalog_tvt_nids.pkl"
)
BLOGCATALOG_SPLIT_SHA256 = (
    "36fefb7c7a3389e96d84b7c48df7822bc6c252097418631e4b5bc1d9757e9a3e"
)
BLOGCATALOG_LABELS_URL = (
    "https://raw.githubusercontent.com/zhao-tong/GAug/"
    "e7d7deab818b822942ba0d20a508ae2cf709a39c/data/graphs/"
    "blogcatalog_labels.pkl"
)
BLOGCATALOG_LABELS_SHA256 = (
    "f0a45d1592df650149112a7bb9728127d8fa52dd5466dab2af496de519170d76"
)
BLOGCATALOG_ADJ_URL = (
    "https://raw.githubusercontent.com/zhao-tong/GAug/"
    "e7d7deab818b822942ba0d20a508ae2cf709a39c/data/graphs/"
    "blogcatalog_adj.pkl"
)
BLOGCATALOG_ADJ_SHA256 = (
    "d154390ee5ac35d5bcafbdc80c0b2ef9fabe460280de7ecf51ea8e5332583b9d"
)
ACTOR_SOURCE_COMMIT = "74ed69daf738ab90a0fb593379311969c8244f39"
ACTOR_SOURCE_URL = (
    "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
    f"{ACTOR_SOURCE_COMMIT}"
)


@dataclass
class Split:
    """One train/validation/test split, as node-index arrays."""

    split_id: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class DatasetBundle:
    """A loaded dataset with its fixed split panel."""

    name: str
    x: torch.Tensor  # (N, F) float
    y: np.ndarray  # (N,) int64, -1 for unlabelled
    edge_index: np.ndarray  # (2, E) int64, as released (may be directed)
    metric: str
    splits: list[Split]
    training_seeds: list[int]
    split_panel: str
    notes: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.y.max()) + 1


def even_quantile_labels(vals: np.ndarray, nclasses: int) -> np.ndarray:
    """Bin a scalar node attribute into even-quantile classes.

    Follows Lim et al. (Non-Homophily-Large-Scale, dataset.py) for the
    arxiv-year label construction.
    """
    label = -np.ones(vals.shape[0], dtype=np.int64)
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        inds = (vals >= lower) & (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    return label


def load_lim_fixed_splits(path: Path) -> list[dict[str, np.ndarray]]:
    """Load the fixed splits released by Lim et al., downloading if absent.

    The file is a pickled list of dicts with keys 'train', 'valid', 'test'
    (node-index arrays), as consumed by load_fixed_splits() in the
    Non-Homophily-Large-Scale repository.
    """
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {ARXIV_YEAR_SPLITS_URL}")
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            urllib.request.urlretrieve(ARXIV_YEAR_SPLITS_URL, temporary)
            if _file_sha256(temporary) != ARXIV_YEAR_SPLITS_SHA256:
                raise RuntimeError(
                    "Downloaded arxiv-year split file does not match the "
                    "registered SHA256 checksum"
                )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    actual_checksum = _file_sha256(path)
    if actual_checksum != ARXIV_YEAR_SPLITS_SHA256:
        raise RuntimeError(
            f"{path}: SHA256 {actual_checksum} differs from the registered "
            f"value {ARXIV_YEAR_SPLITS_SHA256}"
        )
    splits = np.load(path, allow_pickle=True)
    return list(splits)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_verified_file(
    *, url: str, expected_sha256: str, path: Path, description: str
) -> Path:
    """Download a pinned small input if absent and verify its checksum."""
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, temporary)
            if _file_sha256(temporary) != expected_sha256:
                raise RuntimeError(
                    f"Downloaded {description} does not match the registered "
                    "SHA256 checksum"
                )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    actual_checksum = _file_sha256(path)
    if actual_checksum != expected_sha256:
        raise RuntimeError(
            f"{path}: SHA256 {actual_checksum} differs from the registered "
            f"value {expected_sha256}"
        )
    return path


def _mask_panel_to_splits(data, panel_name: str) -> list[Split]:
    """Convert PyG (N, 10) mask tensors into index-array splits."""
    splits = []
    for i in range(data.train_mask.shape[1]):
        splits.append(
            Split(
                split_id=f"{panel_name}-{i}",
                train_idx=data.train_mask[:, i].nonzero().view(-1).numpy(),
                val_idx=data.val_mask[:, i].nonzero().view(-1).numpy(),
                test_idx=data.test_mask[:, i].nonzero().view(-1).numpy(),
            )
        )
    return splits


def load_pubmed(data_root: Path) -> DatasetBundle:
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root=str(data_root / "planetoid"), name="PubMed", split="geom-gcn")
    data = dataset[0]
    return DatasetBundle(
        name="pubmed",
        x=data.x,
        y=data.y.view(-1).numpy().astype(np.int64),
        edge_index=data.edge_index.numpy().astype(np.int64),
        metric="accuracy",
        splits=_mask_panel_to_splits(data, "geomgcn"),
        training_seeds=[1, 2, 3],
        split_panel="geom-gcn-10",
    )


def load_actor(data_root: Path) -> DatasetBundle:
    """Actor/Film with the ten splits released by Geom-GCN."""
    from torch_geometric.datasets import Actor

    class PinnedActor(Actor):
        # PyG otherwise downloads from the mutable ``master`` branch.
        url = ACTOR_SOURCE_URL

    dataset = PinnedActor(root=str(data_root / "actor"))
    data = dataset[0]
    return DatasetBundle(
        name="actor",
        x=data.x.to(torch.float32),
        y=data.y.view(-1).numpy().astype(np.int64),
        edge_index=data.edge_index.numpy().astype(np.int64),
        metric="accuracy",
        splits=_mask_panel_to_splits(data, "geomgcn"),
        training_seeds=[1, 2, 3],
        split_panel="geom-gcn-film-10",
        notes={
            "aliases": ["Actor", "Film"],
            "source": (
                "PyTorch Geometric Actor loader; graph and ten released "
                "Geom-GCN splits (48/32/20 train/validation/test), source "
                f"commit {ACTOR_SOURCE_COMMIT}"
            ),
        },
    )


def _load_blogcatalog_split(path: Path, num_nodes: int) -> tuple[np.ndarray, ...]:
    """Read and validate the checksum-pinned GAug/OpenGSL split."""
    import pickle

    with path.open("rb") as handle:
        raw_split = pickle.load(handle)
    if not isinstance(raw_split, (tuple, list)) or len(raw_split) != 3:
        raise ValueError("BlogCatalog split must contain train, validation, test")
    parts = tuple(
        np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
        for indices in raw_split
    )
    expected_sizes = (519, 1039, 3638)
    if num_nodes == sum(expected_sizes) and tuple(map(len, parts)) != expected_sizes:
        raise ValueError(
            "BlogCatalog split sizes differ from the registered 10/20/70 "
            f"split: expected {expected_sizes}, got {tuple(map(len, parts))}"
        )
    combined = np.concatenate(parts)
    if (
        combined.size != num_nodes
        or np.unique(combined).size != num_nodes
        or not np.array_equal(np.sort(combined), np.arange(num_nodes))
    ):
        raise ValueError("BlogCatalog split does not partition all nodes exactly once")
    return parts


def load_blogcatalog(data_root: Path) -> DatasetBundle:
    """BlogCatalog with the fixed split used by GAug and OpenGSL."""
    import pickle
    from torch_geometric.datasets import AttributedGraphDataset

    dataset = AttributedGraphDataset(
        root=str(data_root / "attributed-graph"), name="BlogCatalog"
    )
    data = dataset[0]
    x = data.x
    if x.layout != torch.strided:
        x = x.to_dense()
    x = x.to(torch.float32)
    if data.y.ndim != 1 or data.y.numel() != data.num_nodes:
        raise ValueError(
            "BlogCatalog must have one class label per node; the loaded "
            f"label shape is {tuple(data.y.shape)}"
        )
    y = data.y.numpy().astype(np.int64)

    auxiliary_root = data_root / "open-gsl-splits"
    split_path = _download_verified_file(
        url=BLOGCATALOG_SPLIT_URL,
        expected_sha256=BLOGCATALOG_SPLIT_SHA256,
        path=auxiliary_root / "blogcatalog_tvt_nids.pkl",
        description="BlogCatalog split",
    )
    labels_path = _download_verified_file(
        url=BLOGCATALOG_LABELS_URL,
        expected_sha256=BLOGCATALOG_LABELS_SHA256,
        path=auxiliary_root / "blogcatalog_labels.pkl",
        description="BlogCatalog label vector",
    )
    adjacency_path = _download_verified_file(
        url=BLOGCATALOG_ADJ_URL,
        expected_sha256=BLOGCATALOG_ADJ_SHA256,
        path=auxiliary_root / "blogcatalog_adj.pkl",
        description="BlogCatalog adjacency matrix",
    )
    # These are pinned, checksum-verified artifacts from the GAug repository.
    with labels_path.open("rb") as handle:
        reference_labels = np.asarray(pickle.load(handle), dtype=np.int64).reshape(-1)
    if not np.array_equal(y, reference_labels):
        raise ValueError(
            "PyG BlogCatalog labels differ from the pinned GAug labels; "
            "the published split cannot safely be applied to this node order"
        )
    with adjacency_path.open("rb") as handle:
        reference_adjacency = pickle.load(handle)
    if reference_adjacency.shape != (data.num_nodes, data.num_nodes):
        raise ValueError("Pinned BlogCatalog adjacency has an unexpected shape")
    from .homophily import to_simple_undirected

    reference_rows, reference_cols = reference_adjacency.nonzero()
    reference_edges = to_simple_undirected(
        np.vstack([reference_rows, reference_cols]), data.num_nodes
    )
    loaded_edges = to_simple_undirected(data.edge_index.numpy(), data.num_nodes)
    if not np.array_equal(reference_edges, loaded_edges):
        raise ValueError(
            "PyG BlogCatalog edges differ from the pinned GAug graph; the "
            "published split cannot safely be applied to this node order"
        )
    train_idx, val_idx, test_idx = _load_blogcatalog_split(
        split_path, data.num_nodes
    )
    if any(np.unique(y[indices]).size != 6 for indices in (
        train_idx,
        val_idx,
        test_idx,
    )):
        raise ValueError("Every BlogCatalog split partition must contain all 6 classes")
    return DatasetBundle(
        name="blogcatalog",
        x=x,
        y=y,
        edge_index=data.edge_index.numpy().astype(np.int64),
        metric="accuracy",
        splits=[
            Split(
                split_id="gaug-public",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
        ],
        training_seeds=list(range(1, 31)),
        split_panel="gaug-opengsl-public-10-20-70",
        notes={
            "source": "PyTorch Geometric AttributedGraphDataset BlogCatalog",
            "splits": (
                "Fixed 10/20/70 train/validation/test split released with "
                "GAug and used as the public BlogCatalog split by OpenGSL; "
                f"split SHA256={BLOGCATALOG_SPLIT_SHA256}; label-order "
                f"SHA256={BLOGCATALOG_LABELS_SHA256}; adjacency "
                f"SHA256={BLOGCATALOG_ADJ_SHA256}"
            ),
            "features": (
                "The released sparse binary attribute matrix is materialised "
                "as float32 because all registered models expect dense input"
            ),
        },
    )


def load_heterophilous(data_root: Path, name: str) -> DatasetBundle:
    """Roman-empire or Amazon-ratings from the Platonov suite."""
    from torch_geometric.datasets import HeterophilousGraphDataset

    dataset = HeterophilousGraphDataset(root=str(data_root / "heterophilous"), name=name)
    data = dataset[0]
    return DatasetBundle(
        name=name.lower(),
        x=data.x,
        y=data.y.view(-1).numpy().astype(np.int64),
        edge_index=data.edge_index.numpy().astype(np.int64),
        metric="accuracy",
        splits=_mask_panel_to_splits(data, "platonov"),
        training_seeds=[1, 2, 3],
        split_panel="platonov-10",
    )


def _load_ogbn_arxiv_raw(data_root: Path):
    from ogb.nodeproppred import PygNodePropPredDataset

    # OGB 1.3.6 calls torch.load on its processed PyG cache; torch>=2.6
    # defaults to weights_only=True, which rejects the PyG classes inside.
    import torch_geometric.data.data as pyg_data
    import torch_geometric.data.storage as pyg_storage

    safe = [pyg_data.Data, pyg_storage.GlobalStorage]
    for attr in ("DataEdgeAttr", "DataTensorAttr"):
        if hasattr(pyg_data, attr):
            safe.append(getattr(pyg_data, attr))
    torch.serialization.add_safe_globals(safe)

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root / "ogb"))
    return dataset, dataset[0]


def load_ogbn_arxiv(data_root: Path) -> DatasetBundle:
    dataset, data = _load_ogbn_arxiv_raw(data_root)
    idx = dataset.get_idx_split()
    split = Split(
        split_id="ogb-time",
        train_idx=idx["train"].numpy().astype(np.int64),
        val_idx=idx["valid"].numpy().astype(np.int64),
        test_idx=idx["test"].numpy().astype(np.int64),
    )
    return DatasetBundle(
        name="ogbn-arxiv",
        x=data.x,
        y=data.y.view(-1).numpy().astype(np.int64),
        edge_index=data.edge_index.numpy().astype(np.int64),
        metric="accuracy",
        splits=[split],
        training_seeds=list(range(1, 31)),
        split_panel="ogb-official",
    )


def load_arxiv_year(data_root: Path) -> DatasetBundle:
    _, data = _load_ogbn_arxiv_raw(data_root)
    years = data.node_year.view(-1).numpy()
    y = even_quantile_labels(years.astype(np.float64), ARXIV_YEAR_NUM_CLASSES)
    raw = load_lim_fixed_splits(data_root / "lim-splits" / "arxiv-year-splits.npy")
    splits = [
        Split(
            split_id=f"lim-{i}",
            train_idx=np.sort(np.asarray(s["train"], dtype=np.int64)),
            val_idx=np.sort(np.asarray(s["valid"], dtype=np.int64)),
            test_idx=np.sort(np.asarray(s["test"], dtype=np.int64)),
        )
        for i, s in enumerate(raw)
    ]
    return DatasetBundle(
        name="arxiv-year",
        x=data.x,
        y=y,
        edge_index=data.edge_index.numpy().astype(np.int64),
        metric="accuracy",
        splits=splits,
        training_seeds=[1, 2, 3, 4, 5, 6],
        split_panel="lim-fixed-5",
        notes={
            "labels": (
                "5-class even-quantile bins of node_year, following "
                "Lim et al. Non-Homophily-Large-Scale dataset.py"
            ),
            "splits": (
                "5 fixed 50/25/25 splits from arxiv-year-splits.npy in the "
                "Non-Homophily-Large-Scale repository (Lim et al.)"
            ),
        },
    )


LOADERS = {
    "pubmed": load_pubmed,
    "ogbn-arxiv": load_ogbn_arxiv,
    "roman-empire": lambda root: load_heterophilous(root, "Roman-empire"),
    "amazon-ratings": lambda root: load_heterophilous(root, "Amazon-ratings"),
    "arxiv-year": load_arxiv_year,
    "actor": load_actor,
    "blogcatalog": load_blogcatalog,
}

DATASET_NAMES = tuple(LOADERS)


def load_dataset(name: str, data_root: Path) -> DatasetBundle:
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    return LOADERS[name](Path(data_root))
