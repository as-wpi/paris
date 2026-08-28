"""Normalise a PARIS result to plain JSON scalars and compare two normalised results.

Every value - float, Series or DataFrame, with whatever index and dtypes - is reduced to nested
lists of ``float`` / ``int`` / ``bool`` / ``str`` / ``None`` before it is written to
``tests/expected/*.json`` and, again, before it is compared. Both sides go through the same
reduction, so dtype and index round-trips never enter the comparison: only labels and numbers do.

Numbers are compared with ``math.isclose(rel_tol=RTOL, abs_tol=ATOL)``. The tolerance is loose
enough to absorb a one-ULP difference in ``pandas`` reductions between environments with and
without the optional ``bottleneck`` accelerator, and BLAS differences across platforms, while
still failing on any change to a convention or formula."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

RTOL = 1e-9
ATOL = 1e-12


def cell(x: Any) -> Any:
    """One JSON scalar for one value: NaN/NaT -> None, +-inf -> "inf"/"-inf", dates -> ISO."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d") if x == x.normalize() else x.isoformat()
    if isinstance(x, pd.Timedelta):
        return str(x)
    if isinstance(x, str):
        return x
    if x is pd.NaT:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, tuple):
        return [cell(v) for v in x]
    return repr(x)


def _labels(index: pd.Index) -> list[Any]:
    return [cell(v) for v in index]


def normalise(value: Any) -> dict[str, Any]:
    """Reduce a function result to the dict that is stored in the expected-values file."""
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "index_name": cell(value.index.name) if value.index.name is not None else None,
            "columns_name": cell(value.columns.name) if value.columns.name is not None else None,
            "index": _labels(value.index),
            "columns": _labels(value.columns),
            "data": [[cell(v) for v in row] for row in value.itertuples(index=False, name=None)],
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "name": cell(value.name) if value.name is not None else None,
            "index_name": cell(value.index.name) if value.index.name is not None else None,
            "index": _labels(value.index),
            "values": [cell(v) for v in value.to_numpy()],
        }
    if isinstance(value, np.ndarray):
        return {"type": "array", "values": [cell(v) for v in value.tolist()]}
    return {"type": "scalar", "value": cell(value)}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    return a == b


def _first_mismatch(actual: list, expected: list, labels: list) -> str:
    for lab, a, e in zip(labels, actual, expected):
        if not _same(a, e):
            return f"at {lab!r}: got {a!r}, expected {e!r}"
    return f"length {len(actual)} vs {len(expected)}"


def assert_matches(actual: Any, expected: dict[str, Any], case_id: str = "") -> None:
    """Fail with a readable message naming the first label whose value differs."""
    got = normalise(actual)
    tag = f"[{case_id}] " if case_id else ""
    assert got["type"] == expected["type"], f"{tag}type {got['type']} != {expected['type']}"
    kind = got["type"]
    if kind == "scalar":
        assert _same(got["value"], expected["value"]), (
            f"{tag}got {got['value']!r}, expected {expected['value']!r}"
        )
        return
    if kind == "array":
        positions = list(range(len(got["values"])))
        assert _same(got["values"], expected["values"]), (
            tag + _first_mismatch(got["values"], expected["values"], positions)
        )
        return
    for key in ("name", "index_name", "columns_name"):
        if key in expected:
            assert got.get(key) == expected[key], (
                f"{tag}{key}: {got.get(key)!r} != {expected[key]!r}"
            )
    assert got["index"] == expected["index"], (
        f"{tag}index labels differ: {got['index'][:5]} vs {expected['index'][:5]}"
    )
    if kind == "Series":
        assert _same(got["values"], expected["values"]), (
            f"{tag}{_first_mismatch(got['values'], expected['values'], got['index'])}"
        )
        return
    assert got["columns"] == expected["columns"], (
        f"{tag}columns differ: {got['columns']} vs {expected['columns']}"
    )
    for row_label, arow, erow in zip(got["index"], got["data"], expected["data"]):
        assert _same(arow, erow), (
            f"{tag}row {row_label!r}: {_first_mismatch(arow, erow, got['columns'])}"
        )
