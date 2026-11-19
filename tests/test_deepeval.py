# test_deepeval.py - DeepEval quality gate over the golden dataset (marker: llm).
#   AI under test: gpt-4o-mini via ask().  Judge: Claude Haiku via DeepEval.
#   Each case counts only if it is RELEVANT (judge score) AND CORRECT (all must_contain words).
#   A wrong answer gets 0.0 even if relevant -> prevents false negatives (wrong answer, green gate).
#   pass_rate(scores, 0.7): 0.7 = threshold per answer; assert rate >= 0.8 = gate for the whole set.
#   Run: python -m pytest -v -s -m llm -k relevancy

import json
import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase
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
        metric = AnswerRelevancyMetric(threshold=0.7, model=JUDGE)
        metric.measure(LLMTestCase(input=case["prompt"], actual_output=answer))

        correct = True
        for word in case["must_contain"]:
            if word not in answer.lower():
                correct = False

        if correct:
            scores[case["id"]] = metric.score
        else:
            scores[case["id"]] = 0.0
            wrong.append(case["id"])
        print(f"{case['id']}: relevancy={metric.score} correct={correct} | {answer}")
    assert wrong == [], f"Incorrect answers: {wrong}"   #Hard gate: the list must be empty; otherwise fail and name the wrong cases
    rate = pass_rate(scores, 0.7)
    assert rate >= 0.8