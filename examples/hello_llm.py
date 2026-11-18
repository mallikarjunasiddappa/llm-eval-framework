from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Describe Stockholm in one sentence.",
    
)

print(response.output_text)
answer = response.output_text
assert "stockholm" in answer.lower()