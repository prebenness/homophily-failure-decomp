"""Check the active Python environment against the reproducibility lock."""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "requirements-reproducibility.txt"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")


def locked_versions(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: expected an exact package pin")
        package, expected = match.groups()
        key = package.lower().replace("_", "-")
        if key in pins:
            raise ValueError(f"{path}:{line_number}: duplicate package {package!r}")
        pins[key] = expected
    return pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    pins = locked_versions(args.lock)
    mismatches: list[dict[str, str | None]] = []
    for package, expected in sorted(pins.items()):
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = None
        if actual != expected:
            mismatches.append(
                {"package": package, "expected": expected, "actual": actual}
            )

    report = {
        "status": "pass" if not mismatches else "fail",
        "python": sys.version.split()[0],
        "locked_packages": len(pins),
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if sys.version.split()[0] != "3.11.15" or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
