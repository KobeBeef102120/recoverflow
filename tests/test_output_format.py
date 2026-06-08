"""Regression tests for output-format classification.

Expected outputs are JSON-serialized in the test payload, which turns tuples
into lists. A correct tuple return value must not be misclassified as an
OUTPUT_FORMAT_ERROR just because the expected value was flattened to a list.
"""
from agent_repair_eval._child_runner import _format_matches, _normalize


def test_tuple_matches_json_flattened_list():
    # Model returns a tuple; expected came back from JSON as a list.
    assert _format_matches((0, 1), [0, 1])
    assert _normalize((0, 1)) == _normalize([0, 1])


def test_nested_tuple_list_equivalence():
    assert _normalize([(1, 2), (3, 4)]) == _normalize([[1, 2], [3, 4]])


def test_numeric_leniency():
    assert _format_matches(1, 1.0)
    assert _format_matches(True, 1)


def test_genuine_format_error_still_caught():
    assert not _format_matches("01", [0, 1])
    assert not _format_matches({"a": 1}, [1])


def test_genuine_value_error_still_caught():
    # Same type, different values — not a format error, must compare unequal.
    assert _format_matches((9, 9), [0, 1])
    assert _normalize((9, 9)) != _normalize([0, 1])
