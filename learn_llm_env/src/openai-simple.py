from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are an eastern poet."},
                                              {"role": "user", "content": """Write a short poem on vscode. 
                                                                            Write the poem in style of haiku.
                                                                            Make sure include a title for the poem.
                                                                            """},
                                          ])

print(response.choices[0].message.content)
