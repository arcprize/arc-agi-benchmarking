import pytest
from arc_agi_testing.utils.parsing import backscan_json_parser

# Test cases for backscan_json_parser

def test_backscan_valid_json_at_end():
    log = "Some preceding text [[1, 2], [3, 4]]"
    expected = [[1, 2], [3, 4]]
    assert backscan_json_parser(log) == expected

def test_backscan_valid_json_only():
    log = "[[1, 2], [3, 4]]"
    expected = [[1, 2], [3, 4]]
    assert backscan_json_parser(log) == expected

def test_backscan_nested_json():
    log = "Log data... [[1, [2, 2]], [3, [4, 4]]]"
    expected = [[1, [2, 2]], [3, [4, 4]]]
    assert backscan_json_parser(log) == expected

def test_backscan_no_json():
    log = "This string has no json content."
    assert backscan_json_parser(log) is None

def test_backscan_incomplete_json_missing_closing():
    log = "Some logs [[1, 2], [3, 4"
    assert backscan_json_parser(log) is None

def test_backscan_incomplete_json_missing_opening():
    # This case might be tricky depending on implementation, assuming it looks for balanced brackets
    log = "Some logs [1, 2], [3, 4]]"
    # The corrected logic finds the last ']', scans back to the matching '[',
    # extracts '[3, 4]', and successfully parses it.
    # HOWEVER, the function must return List[List] or None.
    # Since [3, 4] is not List[List], it should return None.
    assert backscan_json_parser(log) is None

def test_backscan_invalid_json_structure():
    # This case has valid brackets `[]` but invalid JSON inside.
    log = "Log with invalid json [1, 2],[3, 4]]" # Missing outer list brackets, comma is invalid
    # The backscan finds the final ']', scans back to the first '['.
    # Extracts '[1, 2],[3, 4]' which is NOT valid JSON.
    assert backscan_json_parser(log) is None

def test_backscan_empty_string():
    log = ""
    assert backscan_json_parser(log) is None

def test_backscan_only_brackets():
    log = "[]"
    # Parses to [], which is not List[List]. Should return None.
    assert backscan_json_parser(log) is None

def test_backscan_empty_inner_list():
    log = "[[]]"
    expected = [[]]
    assert backscan_json_parser(log) == expected

def test_backscan_multiple_json_blocks():
    log = "First block [1, 2] then some text [[3, 4], [5, 6]]"
    expected = [[3, 4], [5, 6]]
    assert backscan_json_parser(log) == expected

def test_backscan_json_with_internal_brackets():
    log = "Data [annoying] [[1], [2]]"
    expected = [[1], [2]]
    assert backscan_json_parser(log) == expected

def test_backscan_malformed_but_decodable_json():
    # The parser might find the brackets but json.loads fails
    log = "Text [[1, 2], [3, 4]" # Malformed, missing ]
    assert backscan_json_parser(log) is None

def test_backscan_no_closing_bracket_at_all():
    log = "Text [[1, 2], [3, 4"
    assert backscan_json_parser(log) is None

def test_backscan_no_opening_bracket_found():
    log = "Text 1, 2], [3, 4]]"
    # Current logic finds the last ']' and scans back for '['.
    # It should find the matching '[' for the final ']' which starts before 3.
    # Parses to [3, 4]. This is not List[List]. Should return None.
    assert backscan_json_parser(log) is None 