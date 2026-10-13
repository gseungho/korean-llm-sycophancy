import pytest

from ksyco.extract import extract_letter


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B", "B"),
        (" C.\n", "C"),
        ("정답: D", "D"),
        ("답은 B입니다", "B"),
        ("The answer is A.", "A"),
    ],
)
def test_single_choice_letter_is_extracted(text, expected):
    assert extract_letter(text) == expected


@pytest.mark.parametrize("text", ["", "모르겠습니다", "A 또는 B", "ABCD", "Because"])
def test_missing_or_ambiguous_answer_is_a_failure(text):
    assert extract_letter(text) is None
