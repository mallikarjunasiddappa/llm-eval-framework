import pytest
from dotenv import load_dotenv
from openai import OpenAI
from ai_qa.metrics import pass_rate

load_dotenv()
client = OpenAI()

@pytest.mark.llm
def test_capital_of_sweden():
    response = client.responses.create(
        model= "gpt-4o-mini",
        input= "What is the capital of Sweden? Answer in one word."
    )

    answer = response.output_text
    assert "stockholm" in answer.lower()

@pytest.mark.llm
def test_describe_stockholm_quality_gate():
    scores = {}
    for i in range(5):
        response = client.responses.create(
            model= "gpt-4o-mini",
            input= "Describe Stockholm in one sentence, including which country it is in."
        )
        answer = response.output_text
        print(f"run_{i}: {answer}")
        if "sweden" in answer.lower():
            scores[f"run_{i}"] = 1.0
        else:
            scores[f"run_{i}"] = 0.0

    rate = pass_rate(scores, 0.5)
    assert rate >= 0.8
