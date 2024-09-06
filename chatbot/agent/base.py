import base64
import tempfile
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (ConfigurableField, Runnable,
                                      RunnableConfig, RunnableLambda, RunnablePassthrough)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from chatbot.database import vector_db


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
