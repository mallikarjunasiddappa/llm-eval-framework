import pytest
from dotenv import load_dotenv
from openai import OpenAI
from ai_qa.metrics import pass_rate
from ai_qa.llm import ask

load_dotenv()
client = OpenAI()

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
