from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# zero-shot prompting
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are a helpful assistant."},
                                              {"role": "user", "content": """"whats the size of the smallest black hole discovered.
                                                                            """},
                                          ])

print(response.choices[0].message.content)
