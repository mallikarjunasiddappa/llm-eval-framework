# (header description, now also covering test_golden_data_is_valid)

import json
import pytest
from ai_qa.llm import ask


with open("data/golden_set.json") as f:
    CASES = json.load(f)

CASE_IDS = []
for case in CASES:
    CASE_IDS.append(case["id"])


@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_golden_case(case):
    answer = ask(case["prompt"])
    for word in case["must_contain"]:
        assert word in answer.lower()


def test_golden_data_is_valid():
    for case in CASES:
        assert isinstance(case["id"], str)
        assert isinstance(case["prompt"], str)
        assert isinstance(case["must_contain"], list), f"{case['id']}: must_contain must be a list"
        assert isinstance(case["expected_output"], str), f"{case['id']}: expected_output must be text"
        assert case["expected_output"].strip() != "", f"{case['id']}: expected_output is empty"