from argparse import Namespace
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_reverse_workflow_covers_registered_panel_without_training():
    module = runpy.run_path(str(ROOT / "scripts/run_reverse_edge_workflow.py"))
    args = Namespace(data_root=Path("data"), array_root=Path("arrays"),
                     manifest_root=Path("manifests"), graph_root=Path("graphs"),
                     checkpoint_root=Path("checkpoints"), output_root=Path("outputs"),
                     experiment_revision="a" * 40, backend_id="pytorch", device="cpu")
    commands = module["build_commands"](args)
    actual = [command for command in commands
              if "--phase" in command and command[command.index("--phase") + 1] == "actual"]
    assert len(actual) == 120
    combinations = {
        tuple(command[command.index(option) + 1]
              for option in ("--preparation", "--dataset", "--model"))
        for command in actual
    }
    assert len(combinations) == 120
    assert {key[0] for key in combinations} == {"erm", "last", "head", "full"}
    assert all("train" not in Path(command[1]).stem for command in commands)
    assert len([command for command in commands
                if Path(command[1]).name == "run_reverse_edge_inference.py"
                and command[2] == "validate"]) == 4


@pytest.mark.parametrize("complete", [True, False])
def test_reverse_workflow_checks_complete_graphs_before_inference(monkeypatch, complete):
    import analyze_reverse_edge_dose_response as analysis
    import subprocess

    calls = []

    def check_graphs(graph_root):
        calls.append("graph-check")
        if not complete:
            raise analysis.ValidationError("full reverse inference requires the complete accepted graph panel")
        return {}, {}

    monkeypatch.setattr(analysis, "load_graph_contracts", check_graphs)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(Path(command[1]).name))
    module = runpy.run_path(str(ROOT / "scripts/run_reverse_edge_workflow.py"))
    args = ["--experiment-revision", "a" * 40, "--device", "cpu"]
    if complete:
        module["main"](args)
        assert calls.index("graph-check") < calls.index("authenticate_reverse_edge_inputs.py")
        assert "run_reverse_edge_inference.py" in calls
    else:
        with pytest.raises(analysis.ValidationError, match="complete accepted graph panel"):
            module["main"](args)
        assert "authenticate_reverse_edge_inputs.py" not in calls
        assert "run_reverse_edge_inference.py" not in calls
    assert calls[:20] == ["run_reverse_edge_graphs.py"] * 20
