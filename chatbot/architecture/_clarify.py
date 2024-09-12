
import logging
import os
from datetime import datetime
from typing import Annotated, List, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from chatbot.architecture.multiagent import State
from chatbot.tools import get_user_info

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    pending_tool_calls: Annotated[List[AIMessage], add_messages]
    user_info: str

@tool
def cancel_request(reason:str):
    """
    Args:
        reason: reason for cancellation
    """
    return "Exiting from 'clarify requirements mode'. Proceed with the conversation."


class AI:
    def __init__(self, 
                 llm:BaseChatModel, 
                 normal_tools:List[BaseTool], 
                 clarify_tools:List[BaseTool],
                 ) -> None:
        self.llm = llm.bind_tools(normal_tools + clarify_tools)
        self.clarify_tool_names = {t.name for t in clarify_tools}
    def __call__(self, state:State):
        message = (
            ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a helpful assistant for Virtuos Games Company. "
                            " Use the provided tools to assist the user. "
                            "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                            "\nCurrent time: {time}."
                        ),
                        ("placeholder", "{messages}"),
                    ]
                ).partial(time=datetime.now())
            | self.llm
        ).invoke(state)
        if not message.tool_calls:
            return {'messages':[message]}
        updates = {'messages':[],'pending_tool_calls':[]}
        for i, tool_call in enumerate(message.tool_calls):
            if tool_call['name'] in self.clarify_tool_names:
                updates['pending_tool_calls'].append(AIMessage(content="", tool_calls=[message.tool_calls.pop(i)]))
        if message.tool_calls:
            updates['messages'] += [message]
        return updates
    


class ClarifyAI:
    def __init__(self, 
                 llm:BaseChatModel, 
                 normal_tools:List[BaseTool], 
                 clarify_tools:List[BaseTool],
                 ) -> None:
        self.llm = llm.bind_tools(normal_tools + clarify_tools)
        self.clarify_tool_names = {t.name for t in clarify_tools}
        self.llm = llm.bind_tools(normal_tools + clarify_tools + [cancel_request])
    def __call__(self, state:State):
        message = (
            ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a helpful assistant for Virtuos Games Company. "
                            " Your job is to keep asking users to confirm their requirements until you get all args required for {clarify_tool_name}. "
                            " You can suggest args based on your conversation. "
                            " Do not use {clarify_tool_name} unless you get a final confirmation. "
                            " If user no longer wish to continue using {clarify_tool_name} call {cancel_request_tool_name}. "
                            " If user diverged conversation from {clarify_tool_name}, guide them back. "
                            "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                            "\nCurrent time: {time}."
                        ),
                        ("placeholder", "{messages}"),
                    ]
                ).partial(
                    time=datetime.now(),
                    clarify_tool_name=state['pending_tool_calls'][-1].tool_calls[0]['name'],
                    cancel_request_tool_name=cancel_request.name,
                    )
            |self.llm
        ).invoke(state)
        if not message.tool_calls:
            return {'messages':[message]}
        updates = {'messages':[],'pending_tool_calls':[]}
        for i, tool_call in enumerate(message.tool_calls):
            if tool_call['name'] in self.clarify_tool_names:
                if (tool_call['name'] == state['pending_tool_calls'][-1].tool_calls[0]['name']):
                    updates['pending_tool_calls'] += [RemoveMessage(id=state['pending_tool_calls'][-1].id)]
                else:
                    updates['pending_tool_calls'].append(AIMessage(content="", tool_calls=message.tool_calls.pop(i)))
            elif tool_call['name'] == cancel_request.name:
                updates['pending_tool_calls'] += [RemoveMessage(id=state['pending_tool_calls'][-1].id)]

        if message.tool_calls:
            updates['messages'] += [message]
        return updates

def route_ai(state:State):
    return {'messages':[]}

def is_last_message_ai(state:State) -> bool:
    return state['messages'] and isinstance(state['messages'][-1], AIMessage)
def is_pending_tool_call(state:State) -> bool:
    return len(state['pending_tool_calls']) > 0
def is_return_direct(state:State) -> bool:
    messages = state['messages']
    if messages and isinstance(messages[-1], ToolMessage):
        artifact =  messages[-1].artifact
        if artifact and isinstance(artifact, dict) and artifact.get('return_direct'):
            return True
    return False

def route_from_ai(state:State) -> Literal['__end__','tools','clarify_ai']:
    if not is_last_message_ai(state):
        return 'clarify_ai'
    return tools_condition(state)

def route_from_tools(state:State) -> Literal['__end__','ai','clarify_ai']:
    if is_pending_tool_call(state):
        return 'clarify_ai'
    if is_return_direct(state):
        return '__end__'
    return 'ai'
    
def get_builder(llm:BaseChatModel, 
                normal_tools:List[BaseTool], 
                clarify_tools:List[BaseTool],
                ) -> StateGraph:
    builder = StateGraph(State)

    builder.add_edge(START, 'initialise')

    builder.add_node('initialise', lambda state:{'user_info':get_user_info.invoke({"include_session_data_summary":True})})
    builder.add_conditional_edges('initialise', is_pending_tool_call, {True: 'clarify_ai', False:'ai'})

    builder.add_node('ai', AI(llm, normal_tools=normal_tools, clarify_tools=clarify_tools))
    builder.add_conditional_edges('ai', route_from_ai)

    builder.add_node('clarify_ai', ClarifyAI(llm, normal_tools=normal_tools, clarify_tools=clarify_tools))
    builder.add_conditional_edges('clarify_ai', route_from_ai)

    builder.add_node('tools', ToolNode(normal_tools + clarify_tools + [cancel_request]))
    builder.add_conditional_edges('tools', route_from_tools)
    return builder

def get_graph(llm:BaseChatModel, 
              normal_tools:List[BaseTool], 
              clarify_tools:List[BaseTool],
              ):
    memory = MemorySaver()
    builder = get_builder(llm).compile(
        checkpointer=memory, 
        )
    return builder
if __name__ == '__main__':
    from chatbot.mocks import MockChat, mock_tools
    normal_tools = [mock_tools('a')]
    clarify_tools = [mock_tools('b')]
    fname, _ = os.path.splitext(os.path.basename(__file__))
    get_builder(
        MockChat(), normal_tools, clarify_tools
        ).get_graph(xray=True).draw_mermaid_png(output_file_path=f'graphs/{fname}.png')