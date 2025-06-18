from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# chain of thought prompting
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are a mathematician."},
                                              {"role": "user", 
                                               "content": """"
                                                solve this math problem step by step:
                                                If John has 5 apples and gives 2 to Mary, how many does he have left?
                                                """},
                                          ])

print(response.choices[0].message.content)
