""" Only allow one type of clarifying tool at a time,
when cancel called or tool called, pop the latest clarifying tool in queue
when other sensitive tool called, return tool call message with "tool denied, not allowed to use this tool until {clarify_tool_name} requirements are confirmed, or cancelled.
"""
import logging
from functools import partial
from typing import (Annotated, Any, Dict, List, Literal, Set, Type, TypedDict,
                    TypeVar, get_type_hints)

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (Runnable, RunnableConfig, RunnableLambda,
                                      RunnablePassthrough)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from chatbot.architecture.base import return_direct_condition
from chatbot.tools import cancel_clarify_request

logger = logging.getLogger(__name__)


POP = '<pop>'
AGENT = 'agent'
CLARIFY_AGENT = 'clarify_agent'

T = TypeVar('T')
def reducer(left:List[T], right:Literal['<pop>'] | List[T]) -> List[T]:
    if not right:
        return left
    if right == POP:
        return left[:-1]
    return left + right
    
class ClarifyState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    pending_message: AIMessage|None
    pending_tool_calls: Annotated[List[str], reducer]
    current_agent:Literal['main','clarify']


clarify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " Your job is to keep asking users to confirm their requirements until you get all args required for the tools {clarify_tool_name}. "
            " You can suggest args based on your conversation. "
            " Do not use {clarify_tool_name} unless you get a final confirmation. "
            " If user no longer wish to continue using {clarify_tool_name} call {cancel_request_tool_name} to cancel. "
            " If user diverged conversation from {clarify_tool_name}, guide them back. "
        ),
        # ("placeholder", "{messages}"),
    ]
)

def process_pending_message(state:ClarifyState, config:RunnableConfig, sensitive_tool_names:Set[str], cancel_request_tool_name:str=cancel_clarify_request.name):
    pending_message = state["pending_message"]
    updates = {
        'messages':[],
        'pending_message': None,
    }
    # no tool call, happy
    if not pending_message.tool_calls:
        updates['messages'].append(pending_message)
    # message is from main agent
    elif state['current_agent']=='main':
        stc_name = set()
        ntc = []
        for tc in pending_message.tool_calls:
            if tc['name'] == cancel_request_tool_name:
                updates['messages'] += [
                    AIMessage(content="", tool_calls=[tc]), 
                    ToolMessage(content="Nothing to cancel.",tool_call_id=tc['id'])]
            elif tc['name'] in sensitive_tool_names:
                stc_name.add(tc['name'])
            else:
                ntc.append(tc)
        if ntc:
            pending_message.tool_calls = ntc
            updates['messages'].append(pending_message)
        if stc_name:
            updates['pending_tool_calls'] = list(stc_name)
    # message is from clarify agent
    elif state['current_agent']=='clarify':
        pending_tool_name = state['pending_tool_calls'][-1]
        ntc = []
        for tc in pending_message.tool_calls:
            if tc['name'] in [cancel_request_tool_name, pending_tool_name]:
                updates['pending_tool_calls'] = POP
                ntc.append(tc)
            elif tc['name'] in sensitive_tool_names:
                updates['messages'] += [
                    AIMessage(content="", tool_calls=[tc]), 
                    ToolMessage(content=f"Call denied. Complete {pending_tool_name} clarification OR"
                                f" cancel with {cancel_request_tool_name} before using this tool.", tool_call_id=tc['id'])]
            else:
                ntc.append(tc)
        if ntc:
            pending_message.tool_calls = ntc
            updates['messages'].append(pending_message)
            
    return updates

def route_start(state:ClarifyState, config:RunnableConfig):
    return CLARIFY_AGENT if state['pending_tool_calls'] else AGENT

def route_process_pending_message(state:ClarifyState):
    if not state['messages'] or not isinstance(state['messages'][-1], AIMessage):
        assert state['pending_tool_calls']
        return CLARIFY_AGENT
    return tools_condition(state)

def route_tools(state:ClarifyState):
    if state['pending_tool_calls']:
        return CLARIFY_AGENT
    return return_direct_condition(state)



def clarify_agent_builder(
    runnable:Runnable[Any, AIMessage],
    tools:List[BaseTool], 
    sensitive_tools:List[BaseTool],
    *,
    cancel_request_tool:BaseTool=cancel_clarify_request,
    builder:StateGraph | None = None,
    start:str=START,
    end:str=END,
    node_name_prefix:str="",
) -> StateGraph:
    """
    Builds an agent with the ability to handle tool calls, including sensitive ones, and clarifies requests when necessary.

    Args:
        runnable: Must have method .bind_tools implemented. runnable.bind_tools(...).invoke(...) and runnable.bind_tools(...).ainvoke(...) should return BaseMessage.
        tools: List of standard tools available to the agent.
        sensitive_tools: List of tools that require clarification before use.
        cancel_request_tool: A tool that allows cancellation of a clarification request.
        builder: Predefined StateGraph to build upon, or None to create a new one with default state_schema.
        start: Identifier for the start node.
        end: Identifier for the end node.
        node_name_prefix: Prefix for naming nodes within the graph.

    Returns:
        StateGraph: The constructed state graph representing the agent's behavior.
    """
    # validate
    state_schema = builder.schema if builder else ClarifyState
    assert  'messages' in get_type_hints(state_schema), "Key 'messages' is missing from state_schema definition"
    assert  'pending_message' in get_type_hints(state_schema), "Key 'pending_messages' is missing from state_schema definition"
    assert  'pending_tool_calls' in get_type_hints(state_schema), "Key 'pending_tool_calls' is missing from state_schema definition"
    assert hasattr(runnable, "bind_tools"), "runnable.bind_tools not implemented."

    # define agent node
    agent_runnable = runnable.bind_tools(tools + sensitive_tools)
    def invoke(state, config: RunnableConfig):
        return {
            "pending_message": agent_runnable.invoke(state, config),
            "current_agent": "main",
            }
    async def ainvoke(state, config: RunnableConfig):
        return {
            "pending_message": await agent_runnable.ainvoke(state, config),
            "current_agent": "main",
            }
    agent = RunnableLambda(invoke, ainvoke, node_name_prefix + AGENT)

    # define clarify agent node
    clarify_runnable = runnable.bind_tools(tools + sensitive_tools + [cancel_request_tool])
    def clarify_invoke(state:ClarifyState, config: RunnableConfig):
        state['messages'] = clarify_prompt.invoke({
            "cancel_request_tool_name":cancel_request_tool.name,
            'clarify_tool_name':state['pending_tool_calls'][-1],
        }).messages + state['messages']
        return {
            "pending_message":clarify_runnable.invoke(state, config),
            "current_agent": "clarify",
            }
    async def clarify_ainvoke(state:ClarifyState, config: RunnableConfig):
        state['messages'] = clarify_prompt.invoke({
            "cancel_request_tool_name":cancel_request_tool.name,
            'clarify_tool_name':state['pending_tool_calls'][-1],
        }).messages + state['messages']
        return {
            "pending_message":await clarify_runnable.ainvoke(state, config),
            "current_agent": "clarify",
            }
    clarify_agent = RunnableLambda(clarify_invoke, clarify_ainvoke, node_name_prefix + CLARIFY_AGENT)

    # define process_pending_message node
    ppm_node = RunnableLambda(
        func = partial(process_pending_message, 
                       sensitive_tool_names=set([t.name for t in sensitive_tools]), 
                       cancel_request_tool_name=cancel_request_tool.name), 
        name = node_name_prefix + 'process_pending_message')

    # define tools node
    tools_node = ToolNode(tools + sensitive_tools + [cancel_request_tool], name=node_name_prefix + "tools")

    # build
    builder = builder or StateGraph(state_schema)
    # nodes
    builder.add_node(agent.name, agent)
    builder.add_node(clarify_agent.name, clarify_agent)
    builder.add_node(ppm_node.name, ppm_node)
    builder.add_node(tools_node.name, tools_node)
    # edges
    builder.add_conditional_edges(start, route_start, {AGENT:agent.name, CLARIFY_AGENT:clarify_agent.name})
    builder.add_edge(clarify_agent.name, ppm_node.name)
    builder.add_edge(agent.name, ppm_node.name)
    builder.add_conditional_edges(ppm_node.name, route_process_pending_message, {END:end, "tools":tools_node.name, CLARIFY_AGENT:clarify_agent.name})
    builder.add_conditional_edges(tools_node.name, route_tools, {END:end, AGENT:agent.name, CLARIFY_AGENT:clarify_agent.name})
    return builder