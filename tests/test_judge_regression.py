"""
Judge regression test: checks that the groundedness judge still agrees with
the reviewed final labels, so a change in the judge model, DeepEval version
or metric steps is caught before it silently changes test results.
"""
import json
import pytest
from deepeval.test_case import LLMTestCase
from ai_qa.judge import groundedness_metric, measure_with_retry

with open("data/labelled_set.json") as f:
    CASES = json.load(f)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_judge_matches_label(case):
    test_case = LLMTestCase(input=case["question"], actual_output=case["answer"], context=[case["source"]])
    score, attempts = measure_with_retry(groundedness_metric, test_case)
    assert score is not None, f"judge error after {attempts} tries"
    judge = "pass" if score == 1 else "fail"
    assert judge == case["final"]