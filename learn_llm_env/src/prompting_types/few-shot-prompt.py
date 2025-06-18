from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# few shots prompting
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system", "content": "You are a translator"},
        {
            "role": "user",
            "content": """
                Translate these sentences:
                'Hello' -> 'Hola',
                'Goodbye' -> 'Adios'.
                Now translate: 'Thank you".
            """
        },
        {"role":"system", "content": "You are a kids teacher"},
        {
            "role": "user",
            "content": """
                explain complex topics in simple terms:
                1. Example: the concept of light cannot escape a black hole.
                
                Now explain how a car engine works.
            """
        }
    ]
)

print(completion.choices[0].message.content)
