"""Every public function against its frozen result on the sample data (``tests/expected``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import paris
from tests._golden import assert_matches
from tests.cases import BY_MODULE, CASES

EXPECTED = Path(__file__).parent / "expected"
try:
    import scipy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# columns that are NaN without the optional scipy extra (frozen with scipy installed)
SCIPY_ONLY = {"regression_stats": ("alpha_p", "beta_p")}
NOT_FUNCTIONS = {
    "ParisError",
    "GapError",
    "FrequencyError",
    "AlignmentError",
    "Portfolio",
    "ABSOLUTE_METRICS",
    "RELATIVE_METRICS",
    "data",
    "__version__",
}


@pytest.fixture(scope="session")
def expected() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for module in BY_MODULE:
        with (EXPECTED / f"{module}.json").open(encoding="utf-8") as fh:
            out.update(json.load(fh))
    return out


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_matches_expected(case, inputs, expected):
    assert case.id in expected, (
        f"{case.id} has no frozen value: run python -m tests.generate_expected"
    )
    actual, exp = case.run(inputs), expected[case.id]
    if not HAVE_SCIPY and case.fn in SCIPY_ONLY:
        cols = list(SCIPY_ONLY[case.fn])
        actual = actual.drop(columns=cols)
        keep = [i for i, c in enumerate(exp["columns"]) if c not in cols]
        exp = {
            **exp,
            "columns": [exp["columns"][i] for i in keep],
            "data": [[row[i] for i in keep] for row in exp["data"]],
        }
    assert_matches(actual, exp, case.id)


def test_every_public_function_has_a_case():
    public = set(paris.__all__) - NOT_FUNCTIONS
    covered = {c.fn for c in CASES}
    assert public <= covered, f"public functions without a golden case: {sorted(public - covered)}"
    assert covered <= public, f"cases for names that are not public: {sorted(covered - public)}"


def test_no_stale_expected_values(expected):
    ids = {c.id for c in CASES}
    stale = set(expected) - ids
    assert not stale, f"frozen values with no case (regenerate): {sorted(stale)}"
