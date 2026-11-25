import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from ai_qa.llm import ask
from ai_qa.judge import toxicity_metric, bias_metric, pii_metric
from ai_qa.pii import find_pii

with open("data/safety_set.json") as f:
    SAFETY_CASES = json.load(f)

@pytest.mark.llm
@pytest.mark.parametrize("case", SAFETY_CASES, ids=[c["id"] for c in SAFETY_CASES])
def test_safety_case(case):
    answer = ask(case["prompt"])
    print(f"\n{case['id']}: {answer}")
    test_case = LLMTestCase(name=case["id"], input=case["prompt"], actual_output=answer)
    leaked = find_pii(answer)
    assert leaked == [], f"Contact details leaked: {leaked}"
    assert_test(test_case, [toxicity_metric(), bias_metric(), pii_metric()]) 