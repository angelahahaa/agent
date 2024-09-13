
import logging
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import (Annotated, Any, Dict, List, Literal, Set, Type, TypedDict,
                    get_type_hints)

from langchain.tools.base import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (Runnable, RunnableConfig, RunnableLambda,
                                      RunnablePassthrough)
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from chatbot.architecture.multiagent import State
from chatbot.mocks import mock_tool_call
from chatbot.tools import get_user_info

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    
def return_direct_condition(state:State) -> Literal['__end__','agent']:
    message = state['messages'][-1]
    if isinstance(message.artifact, dict) and message.artifact.get('return_direct'):
        return END
    return 'agent'