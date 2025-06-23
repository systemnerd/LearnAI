import asyncio
import os
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

load_dotenv()

def multiply(num1:float, num2:float)->float:
    return num1 * num2

agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers."
)

async def main():
    response = await agent.run("whats 23.54 * 423.54?")
    print(str(response))

if __name__=="__main__":
    asyncio.run(main())
