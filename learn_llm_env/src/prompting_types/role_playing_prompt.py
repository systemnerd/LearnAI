from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Instructional prompting
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are a character in a fantasy novel."},
                                              {"role": "user", 
                                               "content": """"
                                                Describe the setting of the story.
                                                """},
                                          ])

print(response.choices[0].message.content)
