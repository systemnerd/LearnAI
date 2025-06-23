import asyncio
import os
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

def multiply(num1:float, num2:float)->float:
    return num1 * num2

# create a RAG tool
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

async def search_documents(query:str) -> str:
    response = await query_engine.aquery(search_documents)
    return str(response)

agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini"),
    system_prompt="""You are a helpful assistant that can perform calculations and
                     search through documents to answer questions."""
)

async def main():
    # response = await agent.run("whats 23.54 * 423.54?")
    # print(str(response))
    # ctx = Context(agent)
    # response = await agent.run("My name is Nithin", ctx=ctx)
    # response = await agent.run("whats my name", ctx=ctx)
    # print(str(response))
    response = await agent.run(
        "whats the date of the notes? Also, calculate 34.5 *324.4"
    )
    print(str(response))

if __name__=="__main__":
    asyncio.run(main())
