import asyncio

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step
)
from llama_index.utils.workflow import draw_all_possible_flows

class SampleWorkFlow(Workflow):
    @step
    async def step_one(self, ev:StartEvent) -> StopEvent:
        # some custom processing
        return StopEvent(result="Simple result saying Hello world!")

async def main():
    w = SampleWorkFlow(timeout=10,verbose=True)
    results = await w.run()
    print(results)

if __name__=="__main__":
    asyncio.run(main())
    draw_all_possible_flows(SampleWorkFlow, filename="basic_workflow.html")
