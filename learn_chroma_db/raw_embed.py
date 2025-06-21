import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.embeddings.create(
    input="Mars is red in color", model="text-embedding-3-small"
)

print(response.data[0].embedding)
