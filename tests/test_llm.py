import pytest
from dotenv import load_dotenv
from openai import OpenAI

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
