from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# open ended prompting
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are a philosopher."},
                                              {"role": "user", 
                                               "content": """"
                                                what is the meaning of life?
                                                """},
                                          ])

print(response.choices[0].message.content)
