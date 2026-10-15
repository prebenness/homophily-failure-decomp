"""Checkpoint evaluation uses the existing model and metric implementations."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_checkpoints as evaluator
from hfd.training import TrainConfig, train_erm


def tiny_inputs():
    torch.manual_seed(17)
    labels = np.tile(np.arange(2, dtype=np.int64), 12)
    x = torch.randn(24, 4)
    x[:, 0] += torch.from_numpy(labels) * 2
    edges = torch.stack((torch.arange(24), torch.roll(torch.arange(24), 1)))
    fixed = {"labels": labels, "merged_bin": np.tile(np.repeat([0, 1], 2), 6),
             "tail_mask": np.tile([True, True, False, False], 6),
             "split-0.train": np.arange(12), "split-0.val": np.arange(12, 16),
             "split-0.test": np.arange(16, 24)}
    manifest = {"array_checksums_sha256": {}, "local_homophily": {
        "merged_bins": [{"interval": "[0, 0.5)"}, {"interval": "[0.5, 1]"}]}}
    bundle = SimpleNamespace(y=labels, x=x, num_nodes=24, num_classes=2, metric="accuracy")
    return bundle, fixed, manifest, x, edges


def test_training_checkpoint_reload_and_export(tmp_path):
    inputs = tiny_inputs()
    bundle, fixed, _, x, edges = inputs
    model_config = {"z_dim": 8, "head_dropout": 0,
                    "encoder": {"hidden_dim": 8, "num_blocks": 1, "dropout": 0}}
    model = evaluator.build_model("gcn", 4, 2, model_config, num_nodes=24)
    result = train_erm(model, x, edges, torch.from_numpy(bundle.y),
                       torch.from_numpy(fixed["split-0.train"]),
                       torch.from_numpy(fixed["split-0.val"]), TrainConfig(max_epochs=2))
    model.load_state_dict(result.best_state)
    model.eval()
    with torch.no_grad():
        expected = model(x, edges).numpy()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(result.best_state, checkpoint)
    original_checkpoint = checkpoint.read_bytes()
    config = dict(run_id="tiny", dataset="tiny", model="gcn", preparation="erm", split="split-0", seed=1)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "metrics.json").write_text(json.dumps({"checkpoint": {"train_seconds": 1.25},
                                                       "canonicalization": {"version": "old"}}))
    metrics, arrays = evaluator.evaluate_run(tmp_path, config, panel="main", inputs=inputs,
                                              model_config=model_config, parent_root=tmp_path,
                                              device=torch.device("cpu"))
    np.testing.assert_allclose(arrays["logits"], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(arrays["predictions"], expected.argmax(axis=1))
    assert metrics["checkpoint"]["train_seconds"] == 1.25
    assert "canonicalization" not in metrics
    evaluator.save_evaluation(tmp_path, metrics, arrays)
    assert checkpoint.read_bytes() == original_checkpoint
    assert json.loads((tmp_path / "config.json").read_text()) == config
    with np.load(tmp_path / "run_data.npz") as saved:
        np.testing.assert_array_equal(saved["test_idx"], fixed["split-0.test"])
        np.testing.assert_array_equal(saved["predictions"], expected.argmax(axis=1))


def test_evaluation_does_not_require_training_history(tmp_path):
    inputs = tiny_inputs()
    model_config = {"z_dim": 8, "head_dropout": 0,
                    "encoder": {"hidden_dim": 8, "num_blocks": 1, "dropout": 0}}
    model = evaluator.build_model("mlp", 4, 2, model_config, num_nodes=24)
    torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
    config = dict(run_id="tiny", dataset="tiny", model="mlp", preparation="erm", split="split-0", seed=1)
    metrics, _ = evaluator.evaluate_run(tmp_path, config, panel="main", inputs=inputs,
                                        model_config=model_config, parent_root=tmp_path,
                                        device=torch.device("cpu"))
    assert "train_seconds" not in metrics["checkpoint"]
    assert np.isfinite(metrics["tail_macro_f1"])


def test_dfr_reuses_coefficients_without_refitting(tmp_path, monkeypatch):
    inputs = tiny_inputs()
    fixed = inputs[1]
    features = np.column_stack((fixed["labels"], 1 - fixed["labels"])).astype(float)
    classes = np.array([0, 1])
    coef = np.array([[-2., 2.]])
    np.savez(tmp_path / "coefficients.npz", classes=classes, coef=coef, intercept=np.zeros(1),
             scaler_mean=np.zeros(2), scaler_scale=np.ones(2))
    monkeypatch.setattr(evaluator, "check_parent", lambda *args: None)
    monkeypatch.setattr(evaluator, "read_parent_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluator, "extract_classifier_features", lambda **kwargs: (features, 0.))
    config = dict(run_id="tiny", dataset="pubmed", model="gcn", preparation="dfr", split="split-0",
                  seed=1, protocol_id="dfr_l1_joint_balanced_v1")
    metrics, arrays = evaluator.evaluate_run(tmp_path, config, panel="dfr", inputs=inputs,
                                              model_config={}, parent_root=tmp_path, device=torch.device("cpu"))
    fitted = evaluator.DFRFitResult(classes, coef, np.zeros(1), np.zeros(2), np.ones(2), {})
    np.testing.assert_array_equal(arrays["predictions"], fitted.predict(features))
    np.testing.assert_allclose(arrays["probabilities"], fitted.predict_proba(features), rtol=1e-6)
    np.testing.assert_array_equal(arrays["refit_idx"], fixed["split-0.val"][fixed["merged_bin"][fixed["split-0.val"]] >= 0])
    assert metrics["training_objective"]["name"] == "dfr"


@pytest.mark.parametrize("field,value", [("dataset", "wrong"), ("split", "wrong"),
                                          ("fixed_input_checksums", {"labels": "wrong"}),
                                          ("model_config", {"changed": True})])
def test_mismatched_inputs_are_rejected(field, value):
    expected = dict(run_id="tiny", dataset="pubmed", model="gcn", preparation="erm", split="split-0", seed=1)
    config = dict(expected, fixed_input_checksums={}, model_config={})
    config[field] = value
    with pytest.raises(ValueError):
        evaluator.check_config(config, expected, {"array_checksums_sha256": {}}, {}, "main")


def test_panel_sizes_and_ids():
    assert {panel: len(evaluator.panel_runs(panel)) for panel in ("main", "additional", "dfr", "fg")} == {
        "main": 4200, "additional": 1680, "dfr": 90, "fg": 270}
    assert len({row["run_id"] for panel in ("main", "additional", "dfr", "fg")
                for row in evaluator.panel_runs(panel)}) == 6240


def test_wrong_parent_is_rejected(tmp_path):
    config = dict(run_id="tiny", dataset="pubmed", model="gcn", preparation="head",
                  split="split-0", seed=1, parent_run_id="wrong")
    with pytest.raises(ValueError, match="mismatched ERM parent"):
        evaluator.check_parent(config, tmp_path, {"array_checksums_sha256": {}}, {})
