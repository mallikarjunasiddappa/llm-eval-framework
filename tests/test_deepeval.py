# test_deepeval.py - DeepEval quality gate over the golden dataset (marker: llm).
#   AI under test: gpt-4o-mini via ask().  Judge: Claude Haiku via DeepEval.
#   Per case: relevancy (AnswerRelevancyMetric) + correctness (G-Eval vs the reference answer).
#   Correctness = zero tolerance: any case with G-Eval score < 0.7 fails the test and is named.
#   Relevancy  = pass rate: pass_rate(scores, 0.7) must be >= 0.8.
#   Run: python -m pytest -v -s -m llm -k relevancy

import json
import pytest
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from ai_qa.llm import ask
from ai_qa.metrics import pass_rate

with open("data/golden_set.json") as f:
    CASES = json.load(f)

JUDGE = AnthropicModel(model="claude-haiku-5-5")


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
        relevancy  = AnswerRelevancyMetric(threshold=0.7, model=JUDGE)
        relevancy.measure(test_case)

        correctness = GEval(
            name="correctness",
            criteria="Determine whether the actual output is factually consistent with the expected output.",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT
            ],
            model=JUDGE,
            threshold=0.7
        )

        correctness.measure(test_case)
        scores[case["id"]] = relevancy.score
        if correctness.score is None or correctness.score < 0.7:
            wrong.append(case["id"])
        print(f"{case['id']}: relevancy={relevancy.score} correctness={correctness.score} | {answer}")

    assert wrong == [], f"Incorrect answers: {wrong}"
    rate = pass_rate(scores, 0.7)
    assert rate >= 0.8