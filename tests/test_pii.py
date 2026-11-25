"""
Unit tests for the deterministic PII check: clean text finds nothing,
emails and Swedish mobile numbers in common formats are found.
"""
import pytest
from ai_qa.pii import find_pii


@pytest.mark.parametrize("text, expected_count", [
    ("Thank you, Anna Svensson, for being a valued customer!", 0),
    ("Contact anna.svensson@example.com for details.", 1),
    ("Call 070-123 45 67 today.", 1),
    ("Call +46 70 123 45 67 today.", 1),
    ("Mail anna@example.com or call 0701234567.", 2),
    ("Call 070-123 45 67 or 070-765 43 21.", 2),
], ids=["name_only", "email", "phone_dashes", "phone_plus46", "email_and_phone", "two_phones    "])
def test_find_pii(text, expected_count):
    assert len(find_pii(text)) == expected_count