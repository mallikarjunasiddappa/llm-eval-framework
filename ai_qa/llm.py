# llm.py - the ONLY place in the framework that talks to OpenAI.
#   The API key comes from .env (OPENAI_API_KEY), never from code.
#   DEFAULT_MODEL is pinned ("gpt-4o-mini") so results don't change because "latest" moved.
#   ask(prompt, model): sends one prompt (HTTP POST to /responses) and returns only the answer text.
#     Note: API field is input=prompt (prompt= gave a 400 error).
#   Any OpenAI error (400/401/404/429...) is re-raised as RuntimeError naming the model,
#   with the original error attached (from error). Never hides a failure or fakes an answer.

import openai
from dotenv import load_dotenv
from openai import OpenAI
from functools import lru_cache

load_dotenv()

@lru_cache
def get_client():
    return OpenAI()

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = None

def ask(prompt, model=DEFAULT_MODEL, instructions=SYSTEM_PROMPT):
    try:
        response = get_client().responses.create(
            model= model,
            input= prompt,
            instructions=instructions,
        )
    except openai.APIError as error:
        raise RuntimeError(f"LLM call failed for model '{model}': {error}") from error
    return response.output_text
