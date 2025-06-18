from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# temperature prompting
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are a creative writer."},
                                              {"role": "user", 
                                               "content": """"
                                                write a creative tagline for a coffee shop.
                                                """},
                                          ],
                                        #   temperature=0.9, # higher the number more the creative response
                                          top_p=0.1 # controls the diversity of the output.
                                          )

print(response.choices[0].message.content)
