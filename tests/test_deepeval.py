# test_deepeval.py - DeepEval quality gate over the golden dataset (marker: llm).
#   AI under test: gpt-4o-mini via ask().  Judge + thresholds: ai_qa/judge.py.
#   Per case: relevancy (no reference) + G-Eval correctness (vs expected_output).
#   Correctness = zero tolerance: any case below CORRECTNESS_THRESHOLD fails and is named.
#   Relevancy  = soft gate: pass_rate(scores, RELEVANCY_THRESHOLD) must be >= 0.8.
#   Run: python -m pytest -v -s -m llm -k relevancy

import json
import pytest
from deepeval.test_case import LLMTestCase
from ai_qa.llm import ask
from ai_qa.metrics import pass_rate
from ai_qa.judge import (
    relevancy_metric,
    correctness_metric,
    RELEVANCE_THRESHOLD,
    CORRECTNESS_THRESHOLD
    
)

with open("data/golden_set.json") as f:
    CASES = json.load(f)



@pytest.mark.llm
def test_golden_set_relevancy_gate():
    scores = {}
    wrong = []
    for case in CASES:
        answer = ask(case["prompt"])
        test_case = LLMTestCase(
            input=case["prompt"],
            actual_output=answer,
            expected_output=case["expected_output"]
        )
        relevancy=relevancy_metric()
        relevancy.measure(test_case)
        correctness = correctness_metric()
        correctness.measure(test_case)

        scores[case["id"]] = relevancy.score
        if correctness.score is None or correctness.score < CORRECTNESS_THRESHOLD:
            wrong.append(case["id"])
        print(f"{case['id']}: relevancy={relevancy.score} correctness={correctness.score} | {answer}")

    assert wrong == [], f"Incorrect answers: {wrong}"
    rate = pass_rate(scores, RELEVANCE_THRESHOLD)
    assert rate >= 0.8