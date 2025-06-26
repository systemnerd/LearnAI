from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context,
)
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows

class FirstEvent(Event):
    first_output:str

class SecondEvent(Event):
    second_output:str
    response:str

class ProgressEvent(Event):
    msg: str|None

class LLMWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First event is completed")
    
    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        full_response:str|None = ""
        llm = OpenAI(model="gpt-4o-mini")
        generator = await llm.astream_complete(
            prompt="Explain about poetry in two sentences in a witty manner.",
        )

        async for response in generator:
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
            full_response += response.delta
        
        return SecondEvent(
            second_output="Second step complete. full response attached",
            response=str(full_response),
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="step three is happening"))
        print(ev.second_output)
        return StopEvent(result="Workflow complete")
    

async def main():
    w = LLMWorkflow(timeout=10, verbose=False)
    handler = await w.run_stream()

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg)
    
    final_result = await handler
    print("Final result", final_result)

    draw_all_possible_flows(w, filename="basic_workflow.html")

if __name__=="__main__":
    asyncio.run(main())
