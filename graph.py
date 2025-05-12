import os
import tiktoken
import torch
from operator import itemgetter
from typing import Any, Callable, List, Optional, TypedDict, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain_community.tools import TavilySearchResults
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import START, END, StateGraph

from dotenv import load_dotenv
_ = load_dotenv()

from typing import TypedDict, Annotated, Tuple, Dict, List
from langgraph.graph.message import add_messages
from langchain.schema import Document
import operator
from langchain_core.messages import BaseMessage

HF_FOOD_EMBED_MODEL_URL = "https://klnki3w1q88gr09t.us-east-1.aws.endpoints.huggingface.cloud"


class AgentState(TypedDict):
    """State for the foodie-talk langgraph"""
    messages: Annotated[list, add_messages]
    search_query: str
    context: List[Document]
    search_results:Tuple[Union[List[Dict[str, str]], str], Dict]

device_ = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if os.environ.get("HF_FOOD_EMBED_MODEL_URL"):
    embeddings_ = HuggingFaceEndpointEmbeddings(
    model=HF_FOOD_EMBED_MODEL_URL,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    )
else:
    print("HF url not found. Using local model")
    embeddings_ = HuggingFaceEmbeddings(
    model_name="deman539/food-review-ft-snowflake-l-f18eeff6-7504-48c7-af10-1d2d85ca8caa",
    model_kwargs={"device": device_},
    )


def build_graph_chain():
    """Builds a foodie-talk langgraph"""
    openai_chat_model = ChatOpenAI(model="gpt-4.1-mini")
    client = QdrantClient(
    url=os.environ.get('QDRANT_DB_BITTER_MAMMAL'), # Name of the qdrant cluster is bitter_mammal
    api_key=os.environ.get('QDRANT_API_KEY_BITTER_MAMMAL'),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="yelp_reviews",
        embedding=embeddings_,
    )
    qdrant_retriever = vector_store.as_retriever()

    # RAG Prompt
    RAG_PROMPT = """
    CONTEXT:
    {context}

    QUERY:
    {question}

    You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    rag_chain = (
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | rag_prompt | openai_chat_model | StrOutputParser()
    )

    ASSISTANT_PROMPT = """
    You are a foodie assistant. You can answer user's questions about food and restaurants.
    The user may occasionally ask you questions followed by a context which you can rely on to
    answer the questions. If the user has provided context, then use it to answer the question.

    If the question is not related to food or restaurants, you can explain your purpose to the user.
    If the user insists on an unrelated question, you must say "I'm just a foodie-assistant, I can't answer that."
    """
    USER_PROMPT = """
    {question}

    {context}
    """

    ROUTER_PROMPT = """
    You are an intelligent router. You are given below a conversation between a user and an assistant.
    Analyze the conversation and answer based on the following instructions:
    1.Decide if the user's latest message needs additional context gathered from the Internet 
    or a database or restuarant names, locations and reviews. If so, answer ONLY with a SINGLE WORD: "CONTEXT"
    2. Otherwise, answer ONLY with the SINGLE WORD: "ASSISTANT"

    Here is the conversation between the user and the assistant:

    {messages}
    """
    router_prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)

    SEARCH_FORMULATOR_PROMPT = """
    You are an expert at formulating search queries. You are given a history of interactions between
    a user and an AI assistant. Forcus on the most recent user messages and formulate an independent search query
    which can help answer the user's question. Respond ONLY with the search query text.

    Here is the conversation between the user and the assistant:

    {messages}
    """
    search_formulator_prompt = ChatPromptTemplate.from_template(SEARCH_FORMULATOR_PROMPT)

    def router(state: AgentState):
        chat_model = ChatOpenAI(model="gpt-4.1-mini")
        router_chain = router_prompt | chat_model | StrOutputParser()
        router_answer = router_chain.invoke(state["messages"])
        return router_answer

    def search_formulator(state: AgentState):
        chat_model = ChatOpenAI(model="gpt-4.1-mini")
        search_formulator_chain = search_formulator_prompt | chat_model | StrOutputParser()
        search_query = search_formulator_chain.invoke(state["messages"])
        state["search_query"] = search_query
        return state
    
    def search_engine(state: AgentState):
        search_query = state.get("search_query", "")
        if search_query:
            search_tool = TavilySearchResults(max_results=3)
            search_results = search_tool.invoke(search_query)
        state["search_results"] = search_results
        return state
    
    def context_retriever(state: AgentState):
        search_query = state.get("search_query", "")
        if search_query:
            results = rag_chain.invoke({"question": search_query})
            state["context"] = results
        return state


    def assistant(state: AgentState):
        chat_model = ChatOpenAI(model="gpt-4.1-mini")
        assistant_chain = chat_model
        latest_message = state["messages"][-1]
        latest_message = USER_PROMPT.format(question=latest_message.content, context=str(state["context"]) + str(state["search_results"]))
        state["messages"][-1] = latest_message
        response = assistant_chain.invoke(state["messages"])
        return {"messages": [response]}
    
    graph = StateGraph(AgentState)

    graph.add_node("search_formulator", search_formulator)
    graph.add_node("search_engine", search_engine)
    graph.add_node("context_retriever", context_retriever)
    graph.add_node("assistant", assistant)

    graph.add_conditional_edges(START, router, {
        "CONTEXT": "search_formulator",
        "ASSISTANT": "assistant"
    })

    graph.add_edge("search_formulator", "search_engine")
    graph.add_edge("search_engine", "context_retriever")
    graph.add_edge("context_retriever", "assistant")
    graph.add_edge("assistant", END)

    graph = graph.compile()

    messages = [
        AIMessage(content=ASSISTANT_PROMPT),
    ]

    return graph, messages