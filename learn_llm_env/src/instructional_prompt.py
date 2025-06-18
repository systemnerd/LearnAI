from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Instructional prompting
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are a knowledgeable personal trainer."},
                                              {"role": "user", 
                                               "content": """"
                                                Write a 300-word summary of the benefits of exercise, using bullet points.
                                                """},
                                          ])

print(response.choices[0].message.content)
