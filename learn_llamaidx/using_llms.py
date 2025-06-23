import asyncio
from typing import Dict
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool

# sync version
response = OpenAI().complete("Describe San Francisco in two sentences.")
print(response)

async def async_chat():
    response = await OpenAI().acomplete("Describe NYC in two sentences.")
    print(response)

asyncio.run(async_chat())

print("another sentence.. how are you?")

handle = OpenAI().stream_complete("Describe tesls model 3 in two sentences.")
for token in handle:
    print(token.delta, end="", flush=True)

# chat method
llm = OpenAI()
chat_response = llm.chat(messages=[
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Descibe starship in two sentences."),
])
print(str(f"\n{chat_response}"))


# function tooling
def format_response(model_response:str, todays_date, gpt_model) -> Dict:
    """
    Formats the model response into a json dictionary
    todays_date : This has to be todays date
    gpt_model : This should be the same model that we used are chatting
    """
    return {
        "response": model_response,
        "date": todays_date,
        "model": gpt_model,
    }

tool = FunctionTool.from_defaults(fn=format_response)
response = llm.predict_and_call([tool], "Describe solar system in one sentence. Use the tool to format your response")
print(str(f"\n{response}"))
