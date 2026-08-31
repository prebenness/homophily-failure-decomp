"""Verify the declared release inventory of result figures and tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "paper" / "artifacts.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    figures = payload.get("figures")
    tables = payload.get("tables")
    if not isinstance(figures, list) or not isinstance(tables, list):
        raise ValueError(f"{path}: figures and tables must be lists")
    expected_counts = payload.get(
        "expected_counts",
        {"figures": 11, "tables": 13},
    )
    if not isinstance(expected_counts, dict):
        raise ValueError(f"{path}: expected_counts must be an object")
    for key in ("figures", "tables"):
        value = expected_counts.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{path}: expected_counts.{key} must be a non-negative integer"
            )
    payload["expected_counts"] = expected_counts
    return payload


def verify_inventory(
    manifest_path: Path,
    *,
    require_generated: bool = False,
    require_generated_byte_match: bool = False,
) -> dict[str, int]:
    payload = read_manifest(manifest_path)
    errors: list[str] = []
    figure_labels: set[str] = set()
    table_labels: set[str] = set()
    generated_present = 0
    generated_byte_matches = 0
    declared_figure_paths: set[Path] = set()

    for figure in payload["figures"]:
        label = figure.get("label")
        if not isinstance(label, str) or label in figure_labels:
            errors.append(f"duplicate or invalid figure label: {label!r}")
            continue
        figure_labels.add(label)

        artifact_path_text = figure.get("artifact_path")
        if not isinstance(artifact_path_text, str):
            errors.append(f"{label}: missing artifact_path")
            continue
        artifact_path = REPO_ROOT / artifact_path_text
        declared_figure_paths.add(artifact_path.resolve())
        if not artifact_path.is_file():
            errors.append(f"{label}: missing release asset {artifact_path}")
            continue

        expected_hash = figure.get("sha256")
        if expected_hash and file_sha256(artifact_path) != expected_hash:
            errors.append(f"{label}: release asset hash differs")

        generated_path_text = figure.get("generated_path")
        if generated_path_text:
            generated_path = REPO_ROOT / generated_path_text
            if generated_path.is_file():
                generated_present += 1
                if expected_hash and file_sha256(generated_path) == expected_hash:
                    generated_byte_matches += 1
                elif require_generated_byte_match:
                    errors.append(
                        f"{label}: regenerated PDF is not byte-identical to "
                        "the tracked release PDF"
                    )
            elif require_generated:
                errors.append(
                    f"{label}: missing regenerated artifact {generated_path}"
                )

        generator = figure.get("generator")
        if generator and not (REPO_ROOT / generator).is_file():
            errors.append(f"{label}: missing generator {generator}")

    for table in payload["tables"]:
        label = table.get("label")
        if not isinstance(label, str) or label in table_labels:
            errors.append(f"duplicate or invalid table label: {label!r}")
            continue
        table_labels.add(label)

        generated_path_text = table.get("generated_path")
        if not isinstance(generated_path_text, str):
            errors.append(f"{label}: table lacks a generated_path")
        else:
            generated_path = REPO_ROOT / generated_path_text
            if generated_path.is_file():
                generated_present += 1
                label_text = f"\\label{{{label}}}"
                if generated_path.read_text(encoding="utf-8").count(
                    label_text
                ) != 1:
                    errors.append(
                        f"{label}: generated table must contain one label"
                    )
            elif require_generated:
                errors.append(
                    f"{label}: missing regenerated table {generated_path}"
                )

        generator = table.get("generator")
        if generator and not (REPO_ROOT / generator).is_file():
            errors.append(f"{label}: missing generator {generator}")

    figure_directory = REPO_ROOT / "paper" / "figures"
    if figure_directory.is_dir():
        actual_figure_paths = {
            path.resolve()
            for path in figure_directory.rglob("*")
            if path.is_file()
        }
        undeclared_paths = sorted(actual_figure_paths - declared_figure_paths)
        for path in undeclared_paths:
            errors.append(
                "undeclared release figure: "
                f"{path.relative_to(REPO_ROOT).as_posix()}"
            )

    expected_figure_count = payload["expected_counts"]["figures"]
    expected_table_count = payload["expected_counts"]["tables"]
    if len(figure_labels) != expected_figure_count:
        errors.append(
            f"expected {expected_figure_count} released figures; "
            f"found {len(figure_labels)}"
        )
    if len(table_labels) != expected_table_count:
        errors.append(
            f"expected {expected_table_count} data-derived tables; "
            f"found {len(table_labels)}"
        )
    if errors:
        raise RuntimeError("\n".join(errors))

    return {
        "figures": len(figure_labels),
        "tables": len(table_labels),
        "generated_present": generated_present,
        "generated_byte_matches": generated_byte_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-generated", action="store_true")
    parser.add_argument("--require-generated-byte-match", action="store_true")
    args = parser.parse_args()

    counts = verify_inventory(
        args.manifest.resolve(),
        require_generated=args.require_generated,
        require_generated_byte_match=args.require_generated_byte_match,
    )
    print(
        "verified "
        f"{counts['figures']} figures, {counts['tables']} tables, "
        f"{counts['generated_present']} regenerated artifacts"
    )


if __name__ == "__main__":
    main()
