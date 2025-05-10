import os
from typing import List
from chainlit.types import AskFileResponse
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph_chain, AgentState

from dotenv import load_dotenv

_ = load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    graph_chain, initial_messages = build_graph_chain()
    msg = cl.Message(
        content=f"Ready to answer all of your food queries!"
    )
    await msg.send()

    cl.user_session.set("chain", graph_chain)
    cl.user_session.set("messages", initial_messages)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    messages = cl.user_session.get("messages")
    msg = cl.Message(content="")
    messages.append(HumanMessage(content=message.content))
    cl.user_session.set("messages", messages)
    state = AgentState(messages=messages, context=[], search_results=[])
    async for chunk in chain.astream(state, stream_mode="updates"):
        for node, values in chunk.items():
            if node == "assistant":
                messages.append(values["messages"][-1])
                cl.user_session.set("messages", messages)
            await msg.stream_token(f"Receiving update from node: '{node}'\n")
            if values['messages'][-1].content:
                if isinstance(values['messages'][-1], AIMessage):
                    await msg.stream_token(f"{values['messages'][-1].content}\n")

    await msg.send()