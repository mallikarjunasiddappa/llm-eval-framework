"""
Grounded answering: the AI gets a source text in the prompt and must answer
only from it. Checked with relevancy, hallucination (contradictions) and
groundedness (every claim must be in the source).
"""
import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from ai_qa.llm import ask
from ai_qa.judge import relevancy_metric, hallucination_metric, groundedness_metric

with open("data/grounded_set.json") as f:
    GROUNDED_CASES= json.load(f)

def build_prompt(source, question):
    return f"Answer using ONLY the source below.\n\nSource: {source}\n\nQuestion: {question}"


@pytest.mark.llm
@pytest.mark.parametrize("case", GROUNDED_CASES, ids=[c["id"] for c in GROUNDED_CASES])
def test_grounded_case(case):
    answer= ask(build_prompt(case["source"], case["question"]))
    print(f"\n{case['id']}: {answer}")
    test_case = LLMTestCase(
        name=case["id"],
        input=case["question"],
        actual_output=answer,
        context=[case["source"]],
    )
    assert_test(test_case, [relevancy_metric(), hallucination_metric(), groundedness_metric()])
