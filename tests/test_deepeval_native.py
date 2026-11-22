"""
Uses DeepEval's own assert_test: one golden question, judged by
relevancy and correctness. Fails if any metric is below its threshold.
"""

import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from ai_qa.llm import ask
from ai_qa.judge import relevancy_metric, correctness_metric

with open("data/golden_set.json") as f:
    CASES = json.load(f)

@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden_case_native(case):
    answer = ask(case["prompt"])
    test_case = LLMTestCase(
        name=case["id"],
        input=case["prompt"],
        actual_output=answer,
        expected_output=case["expected_output"]
    )

    assert_test(test_case, [relevancy_metric(), correctness_metric()])