from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

llm = OpenAI(model="gpt-4o-mini")

workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools."
)

# finance agent
finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply, add])

finance_workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    llm=OpenAI(model="gpt-4o-mini"),
    tools=finance_tools,
    system_prompt="You are a helpful assistant."
)


async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)

    response = await finance_workflow.run(
        user_msg="show me tesla stock price"
    )
    print(response)

if __name__=="__main__":
    import asyncio
    asyncio.run(main())
