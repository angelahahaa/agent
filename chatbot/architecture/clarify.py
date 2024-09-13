""" Only allow one type of clarifying tool at a time,
when cancel called or tool called, pop the latest clarifying tool in queue
when other sensitive tool called, return tool call message with "tool denied, not allowed to use this tool until {clarify_tool_name} requirements are confirmed, or cancelled.
"""
import logging
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import (Annotated, Any, Dict, List, Literal, Set, Type, TypeVar, TypedDict,
                    get_type_hints)

from langchain.tools.base import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (Runnable, RunnableConfig, RunnableLambda,
                                      RunnablePassthrough)
from langchain_core.tools import BaseTool, tool, ToolCall
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from chatbot.architecture.multiagent import State
from chatbot import mocks
from chatbot.tools import get_user_info, cancel_clarify_request
from chatbot.architecture.base import return_direct_condition

logger = logging.getLogger(__name__)


POP = '<pop>'
AGENT = 'agent'
CLARIFY_AGENT = 'clarify_agent'

T = TypeVar('T')
def reducer(left:List[T], right:Literal['<pop>']|T|List[Literal['<pop>']|T]) -> List[T]:
    if isinstance(right, str):
        right = [right]
    i = len([r for r in right if r==POP])
    assert len(left) <= i
    merged = left[:-i] + [r for r in right if r!=POP]
    return merged
    
class ClarifyState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    pending_message: AIMessage|None
    pending_tool_calls: Annotated[List[Dict], reducer]
    current_agent:Literal['main','clarify']


clarify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " Your job is to keep asking users to confirm their requirements until you get all args required for {clarify_tool_name}. "
            " You can suggest args based on your conversation. "
            " Do not use {clarify_tool_name} unless you get a final confirmation. "
            " If user no longer wish to continue using {clarify_tool_name} call {cancel_request_tool_name} to cancel. "
            " If user diverged conversation from {clarify_tool_name}, guide them back. "
        ),
        ("placeholder", "{messages}"),
    ]
)

def process_pending_message(state:ClarifyState, config:RunnableConfig, sensitive_tool_names:Set[str], cancel_request_tool_name:str=cancel_clarify_request.name):
    pending_message = state["pending_message"]
    updates = {
        'messages':[],
        'pending_message': None,
        'pending_tool_calls': [],
    }
    # no tool call, happy
    if not pending_message.tool_calls:
        updates['messages'].append(pending_message)
    # message is from main agent
    elif state['current_agent']=='main':
        ntc = []
        for tc in pending_message.tool_calls:
            if tc['name'] == cancel_request_tool_name:
                updates['messages'] += [AIMessage(content="", tool_calls=tc), ToolMessage(content="Nothing to cancel.",tool_call_id=tc['id'])]
            elif tc['name'] in sensitive_tool_names:
                updates['pending_tool_calls'].append(tc)
            else:
                ntc.append(tc)
        if ntc:
            pending_message.tool_calls = ntc
            updates['messages'].append(pending_message)
    # message is from clarify agent
    elif state['current_agent']=='clarify':
        pending_tool = state['pending_tool_calls'][-1]['name']
        pop = False
        ntc = []
        for tc in pending_message.tool_calls:
            if tc['name'] == cancel_request_tool_name:
                pop = True
                ntc.append(tc)
            elif tc['name'] == pending_tool:
                pop = True
                ntc.append(tc)
            elif tc['name'] in sensitive_tool_names:
                updates['pending_tool_calls'].append(tc)
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
    if state['pending_message']:
        return CLARIFY_AGENT
    return return_direct_condition(state)



def clarify_agent_builder(
    runnable:Runnable[Any, AIMessage], # llm.bind_tools(tools+sensitive_tools)
    tools:List[BaseTool], 
    sensitive_tools:List[BaseTool],
    cancel_request_tool:BaseTool=cancel_clarify_request,
    state_schema:Type[Dict]|None=None,
    builder:StateGraph|None=None,
    start:str=START,
    end:str=END,
    node_name_prefix:str="",
) -> StateGraph:
    # validate
    if builder is not None and state_schema is not None:
        logger.warning(f"state_schema {state_schema} will be ignored.")
    state_schema = builder.schema if builder else state_schema or ClarifyState
    assert  'messages' in get_type_hints(state_schema), "Key 'messages' is missing from State definition"
    assert  'pending_message' in get_type_hints(state_schema), "Key 'pending_messages' is missing from State definition"
    assert  'pending_tool_calls' in get_type_hints(state_schema), "Key 'pending_tool_calls' is missing from State definition"

    # define variables
    sensitive_tool_names = set([t.name for t in sensitive_tools])

    # define agent
    def invoke(state, config: RunnableConfig):
        return {"pending_message":agent.invoke(state, config)}
    async def ainvoke(state, config: RunnableConfig):
        return {"pending_message":await agent.ainvoke(state, config)}
    agent = RunnableLambda(invoke, ainvoke, node_name_prefix + AGENT)

    # define clarify agent
    clarify_runnable = (
        RunnablePassthrough().assign(**{
            'clarify_tool_name':lambda state: state['pending_tool_calls'][-1],
        }).assign(**{
            'messages':(
                clarify_prompt.partial(cancel_request_tool_name=cancel_request_tool.name) 
                | RunnableLambda(lambda x : x.messages))
        })
        | runnable
    )
    def clarify_invoke(state, config: RunnableConfig):
        return {"pending_message":clarify_runnable.invoke(state, config)}
    async def clarify_ainvoke(state, config: RunnableConfig):
        return {"pending_message":await clarify_runnable.ainvoke(state, config)}

    clarify_agent = RunnableLambda(clarify_invoke, clarify_ainvoke, node_name_prefix + CLARIFY_AGENT)

    # define tools
    tools_node = ToolNode(tools + sensitive_tools + [cancel_request_tool], name=node_name_prefix + "tools")

    builder = StateGraph(state_schema)
    # nodes
    builder.add_node(agent.name, agent)
    builder.add_node(clarify_agent.name, clarify_agent)
    builder.add_node('process_pending_message', 
                     partial(process_pending_message, sensitive_tool_names=sensitive_tool_names, cancel_request_tool_name=cancel_request_tool.name))
    builder.add_node(tools_node.name, tools_node)
    # edges
    builder.add_conditional_edges(start, route_start, {AGENT:agent.name, CLARIFY_AGENT:clarify_agent.name})
    builder.add_edge(agent.name,'process_pending_message')
    builder.add_edge(clarify_agent.name,'process_pending_message')
    builder.add_conditional_edges('process_pending_message', route_process_pending_message, {END:end, "tools":tools_node.name, CLARIFY_AGENT:clarify_agent.name})
    builder.add_conditional_edges('tools', route_tools, {END:end, AGENT:agent.name, CLARIFY_AGENT:clarify_agent.name})
    return builder



if __name__ == '__main__':
    # tests
    tools = [mocks.mock_tool('mock_1'), mocks.mock_tool('mock_2')]
    sensitive_tools = [mocks.mock_tool('sensitive_1'), mocks.mock_tool('sensitive_2')]
    clarify_agent_builder(
        mocks.MockChat().bind_tools(tools+sensitive_tools),tools,sensitive_tools
    ).compile().get_graph(xray=False).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')

    # tests
    def test_process_pending_message(pm, ptcns, clarify_tool_names, expected_reduced_ptcns, expected_message):
        state = ClarifyState(
            pending_message=pm, 
            pending_tool_calls=ptcns)
        result = process_pending_message(state, clarify_tool_names)
        assert result["pending_message"] is None
        actual_reduced_ptcs = reducer(ptcns, result['pending_tool_calls'])
        assert actual_reduced_ptcs == expected_reduced_ptcns, f"{actual_reduced_ptcs} != {expected_reduced_ptcns}"
        if expected_message is None:
            assert result.get('messages') is None, f"{result.get('messages')} != None"
        else:
            assert (sorted([tc['name'] for tc in result['messages'].tool_calls]) == sorted([tc['name'] for tc in expected_message.tool_calls]) \
            and result['messages'].content==expected_message.content), f"{result.get('messages')}!={expected_message}"
        
    def mock_ai_msg(context, tcs):
        return AIMessage(content=context,tool_calls=[mocks.mock_tool_call(tc) for tc in tcs])
    test_process_pending_message(mock_ai_msg("",[]),[], {}, [],mock_ai_msg("",[]))
    test_process_pending_message(mock_ai_msg("meow",["1"]),[], {}, [], mock_ai_msg("meow",["1"]))
    test_process_pending_message(mock_ai_msg("",["1"]),[], {'1'}, ['1'],None)
    test_process_pending_message(mock_ai_msg("",["1"]) , ['1'], {'1'}, [],mock_ai_msg("",["1"]))
    test_process_pending_message(mock_ai_msg("",["1","2"]) , ['1'], {'1'}, [], mock_ai_msg("",["1","2"]))
    test_process_pending_message(mock_ai_msg("",["1","2"]) , ['2', '1'], {'1','2'}, [], mock_ai_msg("",["1","2"]))
    test_process_pending_message(mock_ai_msg("",["1","2"]) , ['1', '3'], {'1'}, ['1', '3', '1'], mock_ai_msg("",["2"]))
    test_process_pending_message(mock_ai_msg("",["3","1","2"]) , ['1', '3'], {'1','3'}, [], mock_ai_msg("",["3","1","2"]))
    test_process_pending_message(mock_ai_msg("",["3","1","2"]) , [], {'1','3'}, ['3','1'], mock_ai_msg("",["2"]))
    test_process_pending_message(mock_ai_msg("meow",["3","1"]) , [], {'1','3'}, ['3','1'], None)
    test_process_pending_message(mock_ai_msg("meow",["3",cancel_clarify_request.name]) , ['1','2'], {'1','2','3'}, ['1','3'], mock_ai_msg("meow",cancel_clarify_request.name))
