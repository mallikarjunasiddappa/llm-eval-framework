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
    CASE = json.load(f)

@pytest.mark.llm
def test_capital_sweden_native():
    question = "What is the capital of Sweden?"
    answer = ask(question)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output="Oslo"
    )

    assert_test(test_case, [relevancy_metric(), correctness_metric()])