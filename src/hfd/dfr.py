"""DFR-style refitting of a linear classifier on frozen representations.

The implementation is deliberately NumPy-only at its boundary.  Test features
are never accepted by :func:`fit_dfr`, so the fitted scaler and classifier
cannot depend on the test set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


PAPER_C_GRID = (1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01)
PROTOCOL = "dfr_l1_joint_balanced_v1"


class DFRSupportError(ValueError):
    """Raised when the requested joint groups cannot support the protocol."""


@dataclass(frozen=True)
class DFRConfig:
    """Configuration for one DFR refit.

    ``random_seed`` controls the joint-stratified split, balanced subsamples,
    and SAGA.  The default C grid is the grid reported by the DFR authors.
    """

    c_grid: tuple[float, ...] = PAPER_C_GRID
    tuning_fraction: float = 0.5
    n_final_fits: int = 10
    random_seed: int = 0
    max_iter: int = 5_000
    tol: float = 1e-4

    def __post_init__(self) -> None:
        c_grid = tuple(float(c) for c in self.c_grid)
        if not c_grid or any((not np.isfinite(c)) or c <= 0 for c in c_grid):
            raise ValueError("c_grid must contain at least one finite positive value")
        if not 0 < self.tuning_fraction < 1:
            raise ValueError("tuning_fraction must lie strictly between zero and one")
        if self.n_final_fits < 1:
            raise ValueError("n_final_fits must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        object.__setattr__(self, "c_grid", c_grid)


@dataclass
class DFRFitResult:
    """Averaged linear classifier and its shared feature standardisation."""

    classes_: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    scaler_mean_: np.ndarray
    scaler_scale_: np.ndarray
    diagnostics: dict

    def _scaled(self, features: np.ndarray) -> np.ndarray:
        x = _validate_features(features, name="features")
        if x.shape[1] != self.coef_.shape[1]:
            raise ValueError(
                f"features have {x.shape[1]} columns; expected {self.coef_.shape[1]}"
            )
        return (x - self.scaler_mean_) / self.scaler_scale_

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        with threadpool_limits(limits=1):
            scaled = self._scaled(features)
            scores = scaled @ self.coef_.T + self.intercept_
        return scores[:, 0] if self.coef_.shape[0] == 1 else scores

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        if scores.ndim == 1:
            positive = _sigmoid(scores)
            return np.column_stack((1.0 - positive, positive))
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable record containing all fitted quantities."""
        return {
            "classes": self.classes_.tolist(),
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_.tolist(),
            "scaler_mean": self.scaler_mean_.tolist(),
            "scaler_scale": self.scaler_scale_.tolist(),
            "diagnostics": self.diagnostics,
        }


def fit_dfr(
    features: np.ndarray,
    labels: np.ndarray,
    homophily_bins: np.ndarray,
    *,
    config: DFRConfig | None = None,
    expected_classes: np.ndarray | None = None,
    expected_bins: np.ndarray | None = None,
) -> DFRFitResult:
    """Fit a DFR classifier on one frozen-representation refit pool.

    C is selected on a deterministic, joint-stratified held-out half using a
    scaler fitted only on the tuning-fit half.  After selection, a separate
    scaler is fitted on the complete refit pool and shared by all final fits.
    The final coefficients and intercepts are averages over
    equal-per-joint-cell balanced samples.
    """

    cfg = config or DFRConfig()
    x, y, bins = _validate_training_arrays(features, labels, homophily_bins)
    classes = _expected_values(y, expected_classes, "class")
    bin_values = _expected_values(bins, expected_bins, "homophily bin")
    cells = tuple((c, b) for c in classes.tolist() for b in bin_values.tolist())

    full_support = _support_counts(y, bins, cells)
    _require_complete_support(full_support, minimum=2, context="complete refit pool")

    fit_idx, tune_idx = joint_stratified_split(
        y,
        bins,
        cells=cells,
        tuning_fraction=cfg.tuning_fraction,
        random_seed=cfg.random_seed,
    )
    fit_support = _support_counts(y[fit_idx], bins[fit_idx], cells)
    tune_support = _support_counts(y[tune_idx], bins[tune_idx], cells)
    _require_complete_support(fit_support, minimum=1, context="tuning-fit partition")
    _require_complete_support(tune_support, minimum=1, context="held-out partition")

    # C selection treats the held-out partition as genuinely held out:
    # preprocessing is estimated on the tuning-fit partition alone.
    with threadpool_limits(limits=1):
        tuning_scaler = StandardScaler().fit(x[fit_idx])
        x_tuning_scaled = tuning_scaler.transform(x)

    tuning_sample, tuning_balance = balanced_joint_sample(
        fit_idx,
        y,
        bins,
        cells=cells,
        random_seed=cfg.random_seed + 1,
    )
    tuning_records = []
    for c_index, c_value in enumerate(cfg.c_grid):
        model, fit_diagnostics = _fit_logistic(
            x_tuning_scaled[tuning_sample],
            y[tuning_sample],
            c_value=c_value,
            random_seed=cfg.random_seed + 100 + c_index,
            config=cfg,
        )
        with threadpool_limits(limits=1):
            prediction = model.predict(x_tuning_scaled[tune_idx])
        cell_accuracies = _joint_cell_accuracies(
            y[tune_idx],
            bins[tune_idx],
            prediction,
            cells,
        )
        tuning_records.append(
            {
                "C": c_value,
                "worst_joint_cell_accuracy": float(min(cell_accuracies.values())),
                "mean_joint_cell_accuracy": float(
                    np.mean(list(cell_accuracies.values()))
                ),
                "overall_accuracy": float(np.mean(prediction == y[tune_idx])),
                "joint_cell_accuracy": _stringify_cell_mapping(cell_accuracies),
                **fit_diagnostics,
            }
        )

    # Worst-cell accuracy is the target.  Mean cell accuracy and stronger
    # regularisation (smaller C) make selection deterministic under ties.
    selected = max(
        tuning_records,
        key=lambda record: (
            record["worst_joint_cell_accuracy"],
            record["mean_joint_cell_accuracy"],
            -record["C"],
        ),
    )
    selected_c = float(selected["C"])

    # Once C has been selected, the complete refit pool is available for the
    # final fit.  Every final model uses this same scaler, so its coefficients
    # occupy a common coordinate system and may be averaged.
    with threadpool_limits(limits=1):
        final_scaler = StandardScaler().fit(x)
        x_final_scaled = final_scaler.transform(x)

    final_coefs = []
    final_intercepts = []
    final_records = []
    for fit_number in range(cfg.n_final_fits):
        sample_seed = cfg.random_seed + 10_000 + fit_number
        sample_idx, balance_count = balanced_joint_sample(
            np.arange(y.size, dtype=np.int64),
            y,
            bins,
            cells=cells,
            random_seed=sample_seed,
        )
        model, fit_diagnostics = _fit_logistic(
            x_final_scaled[sample_idx],
            y[sample_idx],
            c_value=selected_c,
            random_seed=cfg.random_seed + 20_000 + fit_number,
            config=cfg,
        )
        if not np.array_equal(model.classes_, classes):
            raise RuntimeError(
                "final logistic-regression class order differs from audited order"
            )
        final_coefs.append(model.coef_.copy())
        final_intercepts.append(model.intercept_.copy())
        final_records.append(
            {
                "fit_number": fit_number,
                "sample_seed": sample_seed,
                "sample_size": int(sample_idx.size),
                "per_joint_cell": int(balance_count),
                "sample_index_sha256": _index_digest(sample_idx),
                **fit_diagnostics,
            }
        )

    coef = np.mean(np.stack(final_coefs), axis=0)
    intercept = np.mean(np.stack(final_intercepts), axis=0)
    diagnostics = {
        "protocol": PROTOCOL,
        "config": asdict(cfg),
        "random_seed": int(cfg.random_seed),
        "single_cpu_thread": True,
        "solver": "saga",
        "multiclass_adaptation": (
            "multinomial softmax fitted by SAGA with l1_ratio=1.0"
        ),
        "regularisation": "L1",
        "n_refit": int(y.size),
        "n_features": int(x.shape[1]),
        "classes": classes.tolist(),
        "homophily_bins": bin_values.tolist(),
        "joint_cells": [_cell_name(cell) for cell in cells],
        "support": {
            "complete_refit_pool": _stringify_cell_mapping(full_support),
            "tuning_fit_partition": _stringify_cell_mapping(fit_support),
            "held_out_partition": _stringify_cell_mapping(tune_support),
        },
        "split": {
            "fit_n": int(fit_idx.size),
            "held_out_n": int(tune_idx.size),
            "fit_index_sha256": _index_digest(fit_idx),
            "held_out_index_sha256": _index_digest(tune_idx),
        },
        "scaler": {
            "tuning": {
                "fit_scope": "tuning_fit_partition",
                "n_samples_seen": int(tuning_scaler.n_samples_seen_),
                "mean": np.asarray(tuning_scaler.mean_).tolist(),
                "scale": np.asarray(tuning_scaler.scale_).tolist(),
                "held_out_data_used": False,
                "test_data_used": False,
            },
            "final": {
                "fit_scope": "complete_refit_pool",
                "n_samples_seen": int(final_scaler.n_samples_seen_),
                "test_data_used": False,
            },
        },
        "tuning": {
            "criterion": "held-out worst joint-cell accuracy",
            "tie_breakers": [
                "higher mean joint-cell accuracy",
                "smaller C",
            ],
            "balanced_fit_per_joint_cell": int(tuning_balance),
            "balanced_fit_n": int(tuning_sample.size),
            "balanced_fit_index_sha256": _index_digest(tuning_sample),
            "records": tuning_records,
            "selected_C": selected_c,
        },
        "final": {
            "n_fits": int(cfg.n_final_fits),
            "coefficient_aggregation": "arithmetic mean",
            "fits": final_records,
        },
    }
    return DFRFitResult(
        classes_=classes.copy(),
        coef_=coef,
        intercept_=intercept,
        scaler_mean_=np.asarray(final_scaler.mean_, dtype=np.float64),
        scaler_scale_=np.asarray(final_scaler.scale_, dtype=np.float64),
        diagnostics=diagnostics,
    )


def joint_stratified_split(
    labels: np.ndarray,
    homophily_bins: np.ndarray,
    *,
    cells: tuple[tuple[object, object], ...],
    tuning_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every joint cell deterministically into fitting and held-out sets."""

    rng = np.random.default_rng(random_seed)
    fit_parts: list[np.ndarray] = []
    tune_parts: list[np.ndarray] = []
    for class_value, bin_value in cells:
        indices = np.flatnonzero(
            (labels == class_value) & (homophily_bins == bin_value)
        )
        if indices.size < 2:
            raise DFRSupportError(
                f"joint cell {_cell_name((class_value, bin_value))} has "
                f"{indices.size} observation(s); at least two are required"
            )
        shuffled = rng.permutation(indices)
        n_tune = int(np.floor(indices.size * tuning_fraction))
        n_tune = max(1, min(indices.size - 1, n_tune))
        tune_parts.append(shuffled[:n_tune])
        fit_parts.append(shuffled[n_tune:])
    return (
        np.sort(np.concatenate(fit_parts)).astype(np.int64, copy=False),
        np.sort(np.concatenate(tune_parts)).astype(np.int64, copy=False),
    )


def balanced_joint_sample(
    pool_indices: np.ndarray,
    labels: np.ndarray,
    homophily_bins: np.ndarray,
    *,
    cells: tuple[tuple[object, object], ...],
    random_seed: int,
) -> tuple[np.ndarray, int]:
    """Sample an equal number without replacement from every joint cell."""

    pool = np.asarray(pool_indices, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    cell_indices = [
        pool[
            (labels[pool] == class_value)
            & (homophily_bins[pool] == bin_value)
        ]
        for class_value, bin_value in cells
    ]
    balance_count = min(indices.size for indices in cell_indices)
    if balance_count < 1:
        missing = [
            _cell_name(cell)
            for cell, indices in zip(cells, cell_indices)
            if indices.size == 0
        ]
        raise DFRSupportError(
            "balanced sampling pool lacks joint cell(s): " + ", ".join(missing)
        )
    sample = np.concatenate(
        [
            rng.choice(indices, size=balance_count, replace=False)
            for indices in cell_indices
        ]
    )
    return rng.permutation(sample).astype(np.int64, copy=False), int(balance_count)


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    random_seed: int,
    config: DFRConfig,
) -> tuple[LogisticRegression, dict]:
    model = LogisticRegression(
        C=c_value,
        solver="saga",
        l1_ratio=1.0,
        fit_intercept=True,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=random_seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=1):
            model.fit(x, y)
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    return model, {
        "random_seed": int(random_seed),
        "n_iter": np.asarray(model.n_iter_, dtype=int).tolist(),
        "converged": not convergence_messages,
        "convergence_warnings": convergence_messages,
    }


def _validate_training_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    homophily_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _validate_features(features, name="features")
    y = np.asarray(labels)
    bins = np.asarray(homophily_bins)
    if y.ndim != 1 or bins.ndim != 1:
        raise ValueError("labels and homophily_bins must be one-dimensional")
    if x.shape[0] != y.size or y.size != bins.size:
        raise ValueError("features, labels, and homophily_bins must have equal rows")
    if y.size == 0:
        raise ValueError("the refit pool is empty")
    return x, y, bins


def _validate_features(features: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contain non-finite values")
    return x


def _expected_values(
    observed: np.ndarray,
    expected: np.ndarray | None,
    kind: str,
) -> np.ndarray:
    values = np.unique(observed) if expected is None else np.asarray(expected)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"expected {kind} values must be a non-empty vector")
    if np.unique(values).size != values.size:
        raise ValueError(f"expected {kind} values contain duplicates")
    unrecognised = np.setdiff1d(np.unique(observed), values)
    if unrecognised.size:
        raise DFRSupportError(
            f"observed {kind} values absent from expected support: "
            f"{unrecognised.tolist()}"
        )
    return np.sort(values)


def _support_counts(
    labels: np.ndarray,
    bins: np.ndarray,
    cells: tuple[tuple[object, object], ...],
) -> dict[tuple[object, object], int]:
    return {
        cell: int(np.sum((labels == cell[0]) & (bins == cell[1])))
        for cell in cells
    }


def _require_complete_support(
    support: dict[tuple[object, object], int],
    *,
    minimum: int,
    context: str,
) -> None:
    deficient = {
        _cell_name(cell): count
        for cell, count in support.items()
        if count < minimum
    }
    if deficient:
        raise DFRSupportError(
            f"{context} has insufficient joint support; required >= {minimum}: "
            f"{deficient}"
        )


def _joint_cell_accuracies(
    labels: np.ndarray,
    bins: np.ndarray,
    predictions: np.ndarray,
    cells: tuple[tuple[object, object], ...],
) -> dict[tuple[object, object], float]:
    values = {}
    for cell in cells:
        mask = (labels == cell[0]) & (bins == cell[1])
        if not np.any(mask):
            raise DFRSupportError(
                f"evaluation data lack joint cell {_cell_name(cell)}"
            )
        values[cell] = float(np.mean(predictions[mask] == labels[mask]))
    return values


def _cell_name(cell: tuple[object, object]) -> str:
    return f"class={_python_scalar(cell[0])}|bin={_python_scalar(cell[1])}"


def _stringify_cell_mapping(mapping: dict) -> dict[str, object]:
    return {_cell_name(cell): _python_scalar(value) for cell, value in mapping.items()}


def _python_scalar(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _index_digest(indices: np.ndarray) -> str:
    canonical = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    positive = np.empty_like(scores, dtype=np.float64)
    mask = scores >= 0
    positive[mask] = 1.0 / (1.0 + np.exp(-scores[mask]))
    exp_scores = np.exp(scores[~mask])
    positive[~mask] = exp_scores / (1.0 + exp_scores)
    return positive
