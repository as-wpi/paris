"""Regenerate ``tests/expected/<module>.json`` from the current code.

    python -m tests.generate_expected            # every module
    python -m tests.generate_expected risk tables

Run it only when a release intentionally changes numbers, and review the diff: every changed
line is a changed result. It is the only writer of the expected-values files.

It needs the ``scipy`` extra (``uv sync --all-extras``) so that the p-value columns of
``regression_stats`` are frozen as numbers; the tests themselves run with or without scipy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tests._golden import normalise
from tests.cases import BY_MODULE
from tests.conftest import build_inputs

EXPECTED = Path(__file__).parent / "expected"


def main(modules: list[str]) -> None:
    try:
        import scipy  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "generate_expected needs scipy (uv sync --all-extras) so that p-values are frozen"
        ) from e
    inputs = build_inputs()
    EXPECTED.mkdir(exist_ok=True)
    for module in modules or sorted(BY_MODULE):
        out = {c.id: normalise(c.run(inputs)) for c in BY_MODULE[module]}
        path = EXPECTED / f"{module}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, allow_nan=False)
            fh.write("\n")
        print(f"{path.name}: {len(out)} cases")


if __name__ == "__main__":
    main(sys.argv[1:])
