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
    
    # Create a welcome message with elements
    elements = [
        cl.Image(name="foodie", url="https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=800&auto=format&fit=crop&q=60", display="inline"),
        cl.Text(name="welcome", content="👋 Welcome to Foodie Talk! I'm your personal food recommendation assistant. Ask me anything about restaurants, cuisines, or food recommendations!", display="inline")
    ]
    
    msg = cl.Message(
        content="",
        elements=elements
    )
    await msg.send()

    cl.user_session.set("chain", graph_chain)
    cl.user_session.set("messages", initial_messages)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    messages = cl.user_session.get("messages")
    
    # Create a message with loading animation
    msg = cl.Message(content="")
    await msg.send()
    
    # Add thinking animation
    await msg.stream_token("🤔 Thinking...")
    
    messages.append(HumanMessage(content=message.content))
    cl.user_session.set("messages", messages)
    state = AgentState(messages=messages, context=[], search_results=[])
    
    async for chunk in chain.astream(state, stream_mode="updates"):
        for node, values in chunk.items():
            if node == "assistant":
                messages.append(values["messages"][-1])
                cl.user_session.set("messages", messages)
            
            # Update message with node status
            node_emoji = {
                "search_formulator": "🔍",
                "search_engine": "🌐",
                "context_retriever": "📚",
                "assistant": "🍽️"
            }
            
            await msg.stream_token(f"{node_emoji.get(node, '')} + {node.replace('_', ' ').capitalize()} Processing your request...\n")
            
            if values['messages'][-1].content:
                if isinstance(values['messages'][-1], AIMessage):
                    # Format the response with emojis and styling
                    response = values['messages'][-1].content
                    await msg.stream_token(response)

    await msg.send()