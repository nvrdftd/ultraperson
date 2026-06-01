import asyncio
import json
import os
import signal

from dotenv import load_dotenv
from openai import OpenAI

from tools_api import ToolAPIClient

load_dotenv()

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city. Fast response.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. London, Tokyo"
                }
            },
            "required": ["location"]
        }
    },
    {
        "type": "function",
        "name": "research_topic",
        "description": "Research a topic in depth. Takes 3-8 seconds. Use for questions requiring detailed research.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to research, e.g. 'solar energy', 'climate change'"
                }
            },
            "required": ["topic"]
        }
    }
]

async def get_user_input() -> str:
    """Get input from user without blocking the event loop."""
    return await asyncio.to_thread(input, "You: ")

async def call_llm(user_input: str, conversation_history: list, api: ToolAPIClient):
    """Send input to LLM, handle tool calls, yield streaming response."""
    conversation_history.append({
        "role": "user",
        "content": user_input,
    })

    async def generate_stream(client, context):
        """Generate streaming response from LLM."""

        tool_calls = {}
        response_output = []
        stream = None

        try:
            stream = await asyncio.to_thread(
                client.responses.create,
                model=os.environ.get("MODEL", "gpt-5.4-mini"),
                tools=tools,
                instructions="You are an assistant that helps answer questions. You can call tools to get information when needed." \
                "Don't call a tool if you don't need to. If you call a tool, make sure to use the information it returns in your response." \
                "Perhaps you have used the tool before and it is in the conversation history, so check there before calling a tool.",
                input=context,
                stream=True,
            )

            while True:
                event = await asyncio.to_thread(next, stream, None)
                if event is None:
                    break

                # Handle response events
                if event.type == "response.created":
                    yield "Assistant: "
                elif event.type == "response.output_text.delta":
                    response_output.append(event.delta)
                    yield event.delta
                elif event.type == "response.error":
                    yield "Oops, something went wrong. Please try again."
                elif event.type == "response.in_progress":
                    yield "."
                elif event.type == "response.output_item.added":
                    item = event.item
                    if getattr(item, "type", None) == "function_call" and getattr(item, "arguments", None) is None:
                        item.arguments = ""
                    tool_calls[event.output_index] = item
                elif event.type == "response.function_call_arguments.delta":
                    index = event.output_index
                    if index in tool_calls:
                        tool_calls[index].arguments = (tool_calls[index].arguments or "") + event.delta
        except asyncio.CancelledError:
            # Close the SSE stream so the worker thread and TCP socket are released.
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            raise

        # Filter tool calls to only include function calls
        tool_calls = {k: v for k, v in tool_calls.items() if v.type == "function_call"}

        if tool_calls:
            yield "Let me use tools to find the answer...\n"
            tool_calls_context = [conversation_history[-1]]  # Pass the previous user input to tool calls
            await handle_tool_calls(tool_calls, tool_calls_context, api)
            async for chunk in generate_stream(client, tool_calls_context):
                yield chunk

        if response_output:
            conversation_history.append({
                "role": "assistant",
                "content": "".join(response_output),
            })

    client = OpenAI()
    async for chunk in generate_stream(client, conversation_history):
        yield chunk

async def handle_tool_calls(tool_calls: dict, tool_calls_inputs: list, api: ToolAPIClient):

    tool_calls_inputs.extend(list(tool_calls.values()))

    tool_call_outputs = []

    for _, item in tool_calls.items():
        if item.type == "function_call":
            call_output = await call_tool(item.name, json.loads(item.arguments or "{}"), api)
            tool_call_outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": call_output
            })

    tool_calls_inputs.extend(tool_call_outputs)

async def call_tool(tool_name: str, arguments: dict, api: ToolAPIClient) -> str:
    """Call a tool by name with arguments, return output as JSON string."""
    if tool_name == "get_weather":
        location = arguments["location"]
        weather = await get_weather(location, api)
        return json.dumps(weather)

    elif tool_name == "research_topic":
        topic = arguments["topic"]
        research = await research_topic(topic, api)
        return json.dumps(research)

async def get_weather(location: str, api: ToolAPIClient) -> dict:
    """Fetch current weather for a city from the upstream service."""
    return await api.get_weather(location)

async def research_topic(topic: str, api: ToolAPIClient) -> dict:
    """Run an in-depth research call against the upstream service (3-8 s)."""
    return await api.research_topic(topic)

async def main():
    conversation_history = []
    loop = asyncio.get_running_loop()
    turn_task: asyncio.Task | None = None

    def on_sigint() -> None:
        if turn_task and not turn_task.done():
            turn_task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, on_sigint)
    except NotImplementedError:
        pass  # Windows falls back to default KeyboardInterrupt.

    async with ToolAPIClient() as api:
        while True:
            try:
                user_input = await get_user_input()
            except EOFError:
                print()
                break
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            async def _drain_turn() -> None:
                async for chunk in call_llm(user_input, conversation_history, api):
                    print(chunk, end='', flush=True)
                print()

            turn_task = asyncio.create_task(_drain_turn())
            try:
                await turn_task
            except asyncio.CancelledError:
                print("\n[cancelled]")
            finally:
                turn_task = None


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
