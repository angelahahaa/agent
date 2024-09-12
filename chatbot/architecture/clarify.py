
import logging
import os
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Set, Type, TypedDict, get_type_hints

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langchain_core.runnables import (Runnable,
                                      RunnableConfig, RunnableLambda, RunnablePassthrough)
from chatbot.architecture.multiagent import State
from chatbot.tools import get_user_info
from langchain.tools.base import StructuredTool
from chatbot.mocks import mock_tool_call
class ClarifyState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    pending_messages: List[AnyMessage]
    pending_tool_calls: Annotated[List[AIMessage], add_messages]

@tool
def cancel_request(reason:str):
    """
    Args:
        reason: reason for cancellation
    """
    return "Exiting from 'clarify requirements mode'. Proceed with the conversation."

clarify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " Your job is to keep asking users to confirm their requirements until you get all args required for {clarify_tool_name}. "
            " You can suggest args based on your conversation. "
            " Do not use {clarify_tool_name} unless you get a final confirmation. "
            " If user no longer wish to continue using {clarify_tool_name} call {cancel_request_tool_name}. "
            " If user diverged conversation from {clarify_tool_name}, guide them back. "
        ),
        ("placeholder", "{messages}"),
    ]
)

def process_pending_messages(state:ClarifyState, clarify_tool_names:Set[str]):
    pending_messages = state["pending_messages"]
    updates = {
        'messages':[]
    }
    for message in pending_messages:
        clarify_tool_calls = [message.tool_calls[i] for i, tool_call in enumerate(message.tool_calls) if tool_call['name'] in clarify_tool_names]
        if not clarify_tool_calls:
            updates['messages'].append(message)
        else:
            message.tool_calls = [tool_call for tool_call in message.tool_calls if tool_call['name'] not in clarify_tool_names]
            # add the tool call back in if it is one of the pending ones
            clarify_tool_calls_names = set(tc['name'] for tc in clarify_tool_calls)
            ids_to_remove = set(message.id for message in state["pending_tool_calls"] if message.tool_calls[0]['name'] in clarify_tool_calls_names)

def clarify_agent_builder(
    runnable:Runnable[Any, AIMessage],
    tools:List[BaseTool], 
    clarify_tools:List[BaseTool],
    cancel_request_tool:BaseTool=cancel_request,
    state_schema:Type[Dict]=ClarifyState,
) -> StateGraph:
    assert  'messages' in get_type_hints(state_schema), "Key 'messages' is missing from State definition"
    assert  'pending_messages' in get_type_hints(state_schema), "Key 'pending_messages' is missing from State definition"
    assert  'pending_tool_calls' in get_type_hints(state_schema), "Key 'pending_tool_calls' is missing from State definition"
    
    def invoke(state, config: RunnableConfig):
        return {"pending_messages":runnable.invoke(state)}
    async def ainvoke(state, config: RunnableConfig):
        return {"pending_messages":await runnable.ainvoke(state, config)}

    # add the clarifying system prompt to the clarifying agent
    clarify_runnable = (
        RunnablePassthrough().assign(**{
            'clarify_tool_name':lambda state: state['pending_tool_calls'][-1].tool_calls[0]['name'],
        }).assign(**{
            'messages':(
                clarify_prompt.partial(cancel_request_tool_name=cancel_request_tool.name) 
                | RunnableLambda(lambda x : x.messages))
        })
        | runnable
    )

    # print(clarify_runnable.invoke({
    #     "messages":[HumanMessage(content='hi')],
    #     "pending_tool_calls":[AIMessage(content="",tool_calls=[mock_tool_call()])],
    #     "user_info":"123"
    #     }))
    # return 
    def clarify_invoke(state, config: RunnableConfig):
        return {"pending_messages":clarify_runnable.invoke(state)}
    async def clarify_ainvoke(state, config: RunnableConfig):
        return {"pending_messages":await clarify_runnable.ainvoke(state, config)}

def fn(state):
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " {user_info} "
        ),
        ("placeholder", "{messages}"),
    ]
    )
    print('===')
    # print(state)
    print(prompt.invoke(state))
    print('====')
    return AIMessage(content='aha!')
clarify_agent_builder(RunnableLambda(fn),[],[])