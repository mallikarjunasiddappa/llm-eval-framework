# test_llm.py - tests that call the real LLM (marker: llm). Slow and cost a little.
#   Run only these: pytest -m llm      Skip them: pytest -m "not llm"     See prints: add -s
#   test_capital_of_sweden               flexible check: "stockholm" in answer.lower()
#                                        (exact == fails on "Stockholm." with its full stop)
#   test_describe_stockholm_quality_gate same question 5 times; each answer 1.0 if it mentions
#                                        Sweden else 0.0; pass_rate(scores, 0.5) must be >= 0.8.
#                                        0.5 = threshold for ONE answer, 0.8 = target for the set.
#                                        The prompt asks for the country, so check = requirement.
#   test_unknown_model_gives_clear_error non-existent model -> RuntimeError whose message names it

import pytest
from ai_qa.metrics import pass_rate
from ai_qa.llm import ask



@pytest.mark.llm
def test_capital_of_sweden():
    answer = ask("What is the capital of Sweden? Answer in one word.")
    assert "stockholm" in answer.lower()

@pytest.mark.llm
def test_describe_stockholm_quality_gate():
    scores = {}
    for i in range(5):
        answer = ask("Describe Stockholm in one sentence, including which country it is in.")
        print(f"run_{i}: {answer}")
        if "sweden" in answer.lower():
            scores[f"run_{i}"] = 1.0
        else:
            scores[f"run_{i}"] = 0.0

    rate = pass_rate(scores, 0.5)
    assert rate >= 0.8

@pytest.mark.llm
def test_unknown_model_gives_clear_error():
    with pytest.raises(RuntimeError, match="gpt-does-not-exist"):
        ask("Hi", model="gpt-does-not-exist")
