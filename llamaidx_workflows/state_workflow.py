import asyncio
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)

class SetupEvent(Event):
    query: str

class StepTwoEvent(Event):
    query: str

class StatefulFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> SetupEvent | StepTwoEvent:
        db = await ctx.get("some_database", None)
        
        if db is None:
            print("Need to load data")
            return SetupEvent(query=ev.query)
        
        print("Data is present")
        return StepTwoEvent(query=ev.query)
    
    @step
    async def setup(self, ctx: Context, ev: SetupEvent) -> StartEvent:
        await ctx.set("some_database", [1, 2, 3])
        return StartEvent(query=ev.query)

    @step
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        print(await ctx.get("some_database"))
        print(ev.query)
        return StopEvent(result=ev.query)

async def main():
    w = StatefulFlow(timeout=10, verbose=False)
    results = await w.run(query="Start the workflow.")
    print(results)

if __name__=="__main__":
    asyncio.run(main())

