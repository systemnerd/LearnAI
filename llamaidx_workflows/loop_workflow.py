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

class FirstEvent(Event):
    first_output:str

class SecondEvent(Event):
    second_output:str

class LoopEvent(Event):
    loop_output:str

class SampleWorkFlow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        if random.randint(0, 1) == 0:
           print("Bad things happened")
           return LoopEvent(loop_output="Back to step one")
        else:
            print("Good thing happened")
            return FirstEvent(first_output="First step completed") 

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="step_two is done")
    
    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow is done")

async def main():
    w = SampleWorkFlow(timeout=10,verbose=True)
    results = await w.run(first_input="Start the workflow.")
    print(results)

if __name__=="__main__":
    asyncio.run(main())
    draw_all_possible_flows(SampleWorkFlow, filename="basic_workflow.html")
