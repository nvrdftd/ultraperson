import asyncio
from http import client
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

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
    """Get input from user."""
    return input("You: ")

async def call_llm(user_input: str, conversation_history: list):
    """Send input to LLM, handle tool calls, yield streaming response."""
    conversation_history.append({
        "role": "user",
        "content": user_input,
    })

    async def generate_stream(client, context):
        """Generate streaming response from LLM."""

        stream = client.responses.create(
            model=os.environ.get("MODEL", "gpt-5.4-mini"),
            tools=tools,
            instructions="You are an assistant that helps answer questions. You can call tools to get information when needed." \
            "Don't call a tool if you don't need to. If you call a tool, make sure to use the information it returns in your response." \
            "Perhaps you have used the tool before and it is in the conversation history, so check there before calling a tool.",
            input=context,
            stream=True
        )

        tool_calls = {}
        response_output = []

        for event in stream:

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
                tool_calls[event.output_index] = event.item
            elif event.type == "response.function_call_arguments.delta":
                index = event.output_index
                if tool_calls[index]:
                    tool_calls[index].arguments += event.delta

        # Filter tool calls to only include function calls
        tool_calls = {k: v for k, v in tool_calls.items() if v.type == "function_call"}

        if tool_calls:
            yield "Let me use tools to find the answer...\n"
            tool_calls_context = [conversation_history[-1]]  # Pass the previous user input to tool calls
            await handle_tool_calls(tool_calls, tool_calls_context)
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

async def handle_tool_calls(tool_calls: dict, tool_calls_inputs: list):

    tool_calls_inputs.extend(list(tool_calls.values()))

    tool_call_outputs = []

    for _, item in tool_calls.items():
        if item.type == "function_call":
            call_output = await call_tool(item.name, json.loads(item.arguments))
            tool_call_outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": call_output
            })

    tool_calls_inputs.extend(tool_call_outputs)

async def call_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool by name with arguments, return output as string."""
    if tool_name == "get_weather":
        location = arguments["location"]
        weather = await get_weather(location)
        return json.dumps(weather)

    elif tool_name == "research_topic":
        topic = arguments["topic"]
        research = await research_topic(topic)
        return json.dumps(research)

async def get_weather(location: str) -> dict:
    """Fetch weather from API (~200ms)."""
    return {"location": location, "weather": "sunny", "temperature": 25}

async def research_topic(topic: str) -> dict:
    """Research a topic (3-8 seconds). Should be cancellable."""
    return {"topic": topic, "research": "Detailed research on the topic."}

async def main():
    conversation_history = []

    while True:
        user_input = await get_user_input()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # How do you handle cancellation while streaming?
        # client.responses.cancel

        # How do you show pending state during slow tool calls?
        async for chunk in call_llm(user_input, conversation_history):
            print(chunk, end='', flush=True)
        print()
    
    print(conversation_history)

if __name__ == "__main__":
    asyncio.run(main())
