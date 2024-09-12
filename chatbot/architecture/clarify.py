
from collections import defaultdict
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

def reducer(left:List[str], right:str|List[str]):
    if isinstance(right, str):
        right = [right]
    while left and right:
        r = right.pop(0)
        if r == left[-1]:
            left.pop()
        else:
            left.append(r)
            break
    return left + right
    
class ClarifyState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    pending_message: AIMessage|None
    pending_tool_calls: Annotated[List[str], reducer]

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

def process_pending_messages(state:ClarifyState, sensitive_tool_names:Set[str]):
    pending_message = state["pending_message"]
    updates = {
        'pending_message': None,
        'pending_tool_calls': [],
    }
    # Create a mapping of tool names to their indices in the tool calls list
    tool_indices = defaultdict(list)
    num_relevant_calls = 0
    
    # Filter and count relevant tool calls
    for index, tool_call in enumerate(pending_message.tool_calls):
        if tool_call['name'] in sensitive_tool_names:
            tool_indices[tool_call['name']].append(index)
            num_relevant_calls += 1

    # Determine whether to keep the pending message or update it
    if num_relevant_calls == 0:
        updates['messages'] = pending_message
    else:
        # Update pending_tool_calls with relevant tool calls
        for i, name in enumerate(reversed(state['pending_tool_calls'])):
            if i == num_relevant_calls:
                break
            if tool_indices.get(name):
                tool_indices[name].pop()
                updates['pending_tool_calls'].append(name)
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




def clarify_agent_builder(
    runnable:Runnable[Any, AIMessage],
    tools:List[BaseTool], 
    clarify_tools:List[BaseTool],
    cancel_request_tool:BaseTool=cancel_request,
    state_schema:Type[Dict]=ClarifyState,
) -> StateGraph:
    assert  'messages' in get_type_hints(state_schema), "Key 'messages' is missing from State definition"
    assert  'pending_message' in get_type_hints(state_schema), "Key 'pending_messages' is missing from State definition"
    assert  'pending_tool_calls' in get_type_hints(state_schema), "Key 'pending_tool_calls' is missing from State definition"
    
    def invoke(state, config: RunnableConfig):
        return {"pending_message":runnable.invoke(state)}
    async def ainvoke(state, config: RunnableConfig):
        return {"pending_message":await runnable.ainvoke(state, config)}

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
        return {"pending_message":clarify_runnable.invoke(state)}
    async def clarify_ainvoke(state, config: RunnableConfig):
        return {"pending_message":await clarify_runnable.ainvoke(state, config)}

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
# clarify_agent_builder(RunnableLambda(fn),[],[])


if __name__ == '__main__':
    # tests
    def test_process_pending_messages(pending_message, pending_tcs, clarify_tool_names, out_pending_tc_names, out_message):
        state = ClarifyState(
            pending_message=pending_message, 
            pending_tool_calls=pending_tcs)
        result = process_pending_messages(state, clarify_tool_names)
        assert result["pending_message"] is None
        assert result['pending_tool_calls'] == out_pending_tc_names, f"{result['pending_tool_calls']} != {out_pending_tc_names}"
        assert (out_message is None and result.get('messages') is None) or \
            (sorted([tc['name'] for tc in result['messages'].tool_calls]) == sorted([tc['name'] for tc in out_message.tool_calls]) \
            and result['messages'].content==out_message.content), f"{result.get('messages')}!={out_message}"
        
    def mock_ai_msg(context, tcs):
        return AIMessage(content=context,tool_calls=[mock_tool_call(tc) for tc in tcs])
    test_process_pending_messages(mock_ai_msg("",[]),[], {}, [],mock_ai_msg("",[]))
    test_process_pending_messages(mock_ai_msg("meow",["1"]),[], {}, [], mock_ai_msg("meow",["1"]))
    test_process_pending_messages(mock_ai_msg("",["1"]),[], {'1'}, ['1'],None)
    test_process_pending_messages(mock_ai_msg("",["1"]) , ['1'], {'1'}, ["1"],mock_ai_msg("",["1"]))
    test_process_pending_messages(mock_ai_msg("",["1","2"]) , ['1'], {'1'}, ["1"], mock_ai_msg("",["1","2"]))
    test_process_pending_messages(mock_ai_msg("",["1","2"]) , ['2', '1'], {'1','2'}, ['1','2'], mock_ai_msg("",["1","2"]))
    test_process_pending_messages(mock_ai_msg("",["1","2"]) , ['1', '3'], {'1'}, ["1"], mock_ai_msg("",["2"]))
    test_process_pending_messages(mock_ai_msg("",["3","1","2"]) , ['1', '3'], {'1','3'}, ['3','1'], mock_ai_msg("",["3","1","2"]))
    test_process_pending_messages(mock_ai_msg("",["3","1","2"]) , [], {'1','3'}, ['3','1'], mock_ai_msg("",["2"]))
    test_process_pending_messages(mock_ai_msg("meow",["3","1"]) , [], {'1','3'}, ['3','1'], None)
