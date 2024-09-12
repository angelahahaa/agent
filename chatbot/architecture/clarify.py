
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
from chatbot import mocks
from chatbot.tools import get_user_info, cancel_clarify_request
from chatbot.architecture.base import return_direct_condition
POP = '<pop>'
AGENT = 'agent'
CLARIFY_AGENT = 'clarify_agent'

def reducer(left:List[str], right:str|List[str]):
    # print(f"left: {left}, right: {right}")
    if isinstance(right, str):
        right = [right]
    i = len([r for r in right if r==POP])
    merged = left[:-i] + [r for r in right if r!=POP]
    # print(f"merged: {merged}")
    return merged
    
class ClarifyState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    pending_message: AIMessage|None
    pending_tool_calls: Annotated[List[str], reducer]


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

def process_pending_messages(state:ClarifyState, sensitive_tool_names:Set[str], cancel_request_tool_name:str=cancel_clarify_request.name):
    pending_message = state["pending_message"]
    updates = {
        'pending_message': None,
        'pending_tool_calls': [],
    }
    # Create a mapping of tool names to their indices in the tool calls list
    tool_indices = defaultdict(list)
    num_sensitive_calls = 0
    has_cancel_request_tool_name = False
    
    # Filter and count relevant tool calls
    for index, tool_call in enumerate(pending_message.tool_calls):
        if tool_call['name'] in sensitive_tool_names:
            tool_indices[tool_call['name']].append(index)
            num_sensitive_calls += 1
        if tool_call['name'] == cancel_request_tool_name:
            has_cancel_request_tool_name = True

    # Determine whether to keep the pending message or update it
    if has_cancel_request_tool_name:
        updates['pending_tool_calls'].append(POP)
    if num_sensitive_calls == 0:
        updates['messages'] = pending_message
    else:
        # Update pending_tool_calls with relevant tool calls
        for i, name in enumerate(reversed(state['pending_tool_calls'])):
            if i==0 and has_cancel_request_tool_name:
                # has_cancel_request_tool_name means the last tool call is already popped
                pass
            if i == num_sensitive_calls:
                break
            if tool_indices.get(name):
                tool_indices[name].pop()
                updates['pending_tool_calls'].append(POP)
            else:
                break
        
        # Add remaining relevant tool calls to pending_tool_calls
        for indices in tool_indices.values():
            updates['pending_tool_calls'].extend(
                pending_message.tool_calls[index]['name'] for index in indices
            )
        
        # Remove processed tool calls from pending_message
        pending_message.tool_calls = [
            tool_call for i, tool_call in enumerate(pending_message.tool_calls)
            if i not in tool_indices[tool_call['name']]
        ]
        if pending_message.tool_calls:
            updates['messages'] = pending_message
    return updates

def route_process_pending_messages(state:ClarifyState):
    if not state['messages'] or not isinstance(state['messages'][-1], AIMessage):
        assert state['pending_tool_calls']
        return CLARIFY_AGENT
    return tools_condition(state)

def route_tools(state:ClarifyState):
    if state['pending_message']:
        return CLARIFY_AGENT
    return return_direct_condition(state)



def clarify_agent_builder(
    runnable:Runnable[Any, AIMessage],
    clarify_runnable:Runnable[Any, AIMessage], # bind this one with cancel_request_tool
    tools:List[BaseTool], 
    sensitive_tools:List[BaseTool],
    cancel_request_tool:BaseTool=cancel_clarify_request,
    state_schema:Type[Dict]=ClarifyState,
) -> StateGraph:
    assert  'messages' in get_type_hints(state_schema), "Key 'messages' is missing from State definition"
    assert  'pending_message' in get_type_hints(state_schema), "Key 'pending_messages' is missing from State definition"
    assert  'pending_tool_calls' in get_type_hints(state_schema), "Key 'pending_tool_calls' is missing from State definition"

    sensitive_tool_names = set([t.name for t in sensitive_tools])
    
    def invoke(state, config: RunnableConfig):
        return {"pending_message":runnable.invoke(state, config)}
    async def ainvoke(state, config: RunnableConfig):
        return {"pending_message":await runnable.ainvoke(state, config)}

    # add the clarifying system prompt to the clarifying agent
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

    agent = RunnableLambda(invoke, ainvoke, AGENT)
    clarify_agent = RunnableLambda(clarify_invoke, clarify_ainvoke, CLARIFY_AGENT)

    builder = StateGraph(state_schema)
    builder.add_node(AGENT, agent)
    builder.add_node(CLARIFY_AGENT, clarify_agent)
    builder.add_node('process_pending_messages', lambda state: process_pending_messages(state,sensitive_tool_names=sensitive_tool_names, cancel_request_tool_name=cancel_request_tool.name))
    builder.add_node('tools', ToolNode(tools + sensitive_tools + [cancel_request_tool]))
    
    builder.add_conditional_edges(START, lambda state:CLARIFY_AGENT if state['pending_tool_calls'] else AGENT, [AGENT, CLARIFY_AGENT])
    builder.add_edge(AGENT,'process_pending_messages')
    builder.add_edge(CLARIFY_AGENT,'process_pending_messages')
    builder.add_conditional_edges('process_pending_messages', route_process_pending_messages, [END, "tools", CLARIFY_AGENT])
    builder.add_conditional_edges('tools', route_tools, {END:END, 'agent':AGENT, CLARIFY_AGENT:CLARIFY_AGENT})
    return builder



if __name__ == '__main__':
    # tests
    tools = [mocks.mock_tool('mock_1'), mocks.mock_tool('mock_2')]
    sensitive_tools = [mocks.mock_tool('sensitive_1'), mocks.mock_tool('sensitive_2')]
    clarify_agent_builder(
        mocks.MockChat().bind_tools(tools+sensitive_tools),tools,sensitive_tools
    ).compile().get_graph(xray=False).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')

    # tests
    def test_process_pending_messages(pm, ptcns, clarify_tool_names, expected_reduced_ptcns, expected_message):
        state = ClarifyState(
            pending_message=pm, 
            pending_tool_calls=ptcns)
        result = process_pending_messages(state, clarify_tool_names)
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
    test_process_pending_messages(mock_ai_msg("",[]),[], {}, [],mock_ai_msg("",[]))
    test_process_pending_messages(mock_ai_msg("meow",["1"]),[], {}, [], mock_ai_msg("meow",["1"]))
    test_process_pending_messages(mock_ai_msg("",["1"]),[], {'1'}, ['1'],None)
    test_process_pending_messages(mock_ai_msg("",["1"]) , ['1'], {'1'}, [],mock_ai_msg("",["1"]))
    test_process_pending_messages(mock_ai_msg("",["1","2"]) , ['1'], {'1'}, [], mock_ai_msg("",["1","2"]))
    test_process_pending_messages(mock_ai_msg("",["1","2"]) , ['2', '1'], {'1','2'}, [], mock_ai_msg("",["1","2"]))
    test_process_pending_messages(mock_ai_msg("",["1","2"]) , ['1', '3'], {'1'}, ['1', '3', '1'], mock_ai_msg("",["2"]))
    test_process_pending_messages(mock_ai_msg("",["3","1","2"]) , ['1', '3'], {'1','3'}, [], mock_ai_msg("",["3","1","2"]))
    test_process_pending_messages(mock_ai_msg("",["3","1","2"]) , [], {'1','3'}, ['3','1'], mock_ai_msg("",["2"]))
    test_process_pending_messages(mock_ai_msg("meow",["3","1"]) , [], {'1','3'}, ['3','1'], None)
    test_process_pending_messages(mock_ai_msg("meow",["3",cancel_clarify_request.name]) , ['1','2'], {'1','2','3'}, ['1','3'], mock_ai_msg("meow",cancel_clarify_request.name))
