import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DEFAULT_MODEL = "gpt-4o-mini"

def ask(prompt, model=DEFAULT_MODEL):
    try:
        response = client.responses.create(
            model= model,
            input= prompt,
        )
    except openai.APIError as error:
        raise RuntimeError(f"LLM call failed for model '{model}': {error}") from error
    return response.output_text