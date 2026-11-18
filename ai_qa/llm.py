from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DEFAULT_MODEL = "gpt-4o-mini"

def ask(prompt, model=DEFAULT_MODEL):
    response = client.responses.create(
        model= model,
        input= prompt,
    )
    return response.output_text