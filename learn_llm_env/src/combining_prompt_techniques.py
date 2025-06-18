from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# combining prompting techniques
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[
                                              {"role": "system", "content": "You are a travel blogger."},
                                              {"role": "user", 
                                               "content": """"
                                                Write a 100 word blog post about your recent trip to New York.
                                                Make sure to give a step by step itenary of the trip.
                                                """},
                                          ],
                                          temperature=0.9, # higher the number more the creative response
                                        #   top_p=0.1 # controls the diversity of the output.
                                        stream=True
                                          )

# print(response.choices[0].message.content)
for chunk in response:
    if chunk.choices[0].delta is not None:
        print(chunk.choices[0].delta.content, end="")
    print("\n")
