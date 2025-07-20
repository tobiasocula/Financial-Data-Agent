from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from langchain_core.messages import BaseMessage, messages_to_dict, messages_from_dict
import json
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from logger import logger
from agent_config import graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentMemoryManager:
    def __init__(self, path: str = "memory.json"):
        self.path = path

    def save(self, messages: list[BaseMessage]) -> None:
        """Save list of BaseMessage to JSON."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(messages_to_dict(messages), f, indent=2)

    def load(self) -> list[BaseMessage]:
        """Load message history from JSON file."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return messages_from_dict(data)

    def clear(self):
        """Reset memory."""
        if os.path.exists(self.path):
            os.remove(self.path)

memory = AgentMemoryManager("memory.json")

# Load past messages
past_messages = memory.load()

class PromptRequest(BaseModel):
    input: str

@app.post("/prompt")
def prompt(input: PromptRequest):
    past_messages = memory.load()
    messages = [SystemMessage(
        content="You are a helpful AI assistant. Do not repeat or echo the user's message. Just answer clearly."
    )] + past_messages + [HumanMessage(content=input.input)]

    def event_generator():

        """
        What messages are in state["messages"]?
        HumanMessage: User’s message (content is a string: what the user wrote)

        AIMessage: LLM’s response (content is usually a string: generated text reply)

        ToolMessage: Tool/function result (content is the return value of your tool function—this can be a string, number, dict, etc.)

        All these are subclasses of a message base class, and each has a .content attribute.
        """
        inputs = {"messages": messages}
        for state in graph.stream(inputs, stream_mode="values"):
            latest = state["messages"][-1]
            # for msg in state["messages"]:
            #     logger.info(f"[DEBUG MSG] {repr(msg)}", extra={"component": "POST"})
            if latest.content == input.input:
                continue

            yield json.dumps({
                "role": getattr(latest, "type", "unknown"),
                "type": latest.type,
                "content": latest.content
                }) + "\n"

        # Save full updated memory
        memory.save(state["messages"])

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/clear-memory")
def clear_memory():
    logger.info('called clear-memory (endpoint)', extra={'component': 'POST'})
    memory.clear()