import asyncio
import random

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event
)
from llama_index.utils.workflow import draw_all_possible_flows

class BranchA1Event(Event):
    payload: str

class BranchA2Event(Event):
    payload: str

class BranchB1Event(Event):
    payload: str

class BranchB2Event(Event):
    payload: str

class BranchWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchA1Event | BranchB1Event:
        if random.randint(0, 1) == 0:
            print("Go to Branch A")
            return BranchA1Event(payload="Branch A")
        else:
            print("Go to Branch B")
            return BranchB1Event(payload="Branch B")
        
    @step
    async def step_a1(self, ev: BranchA1Event) -> BranchA2Event:
        print(ev.payload)
        return BranchA2Event(payload=ev.payload)
    
    @step
    async def step_a2(self, ev: BranchA2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch A complete.")
    
    @step
    async def step_b1(self, ev: BranchB1Event) -> BranchB2Event:
        print(ev.payload)
        return BranchB2Event(payload=ev.payload)
    
    @step
    async def step_b2(self, ev: BranchB2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch B complete")

async def main():
    w = BranchWorkflow(timeout=10,verbose=False)
    results = await w.run(first_input="Start the workflow")
    print(results)

if __name__=="__main__":
    asyncio.run(main())
    draw_all_possible_flows(BranchWorkflow, filename="basic_workflow.html")
