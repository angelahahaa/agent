from collections import defaultdict
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableConfig, RunnablePassthrough, RunnableLambda)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Set
import base64
import tempfile
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, TypedDict, NotRequired

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (ConfigurableField, Runnable,
                                      RunnableConfig, RunnableLambda, RunnablePassthrough)
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from chatbot.database import vector_db
from chatbot.agent.search import SearchAssistant
from typing import Optional, Callable
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                    RemoveMessage, SystemMessage, ToolMessage)

from langchain.tools.base import StructuredTool
import random
from uuid import uuid4
import logging
logger = logging.getLogger()

def _fake_tool(name:str) -> BaseTool:
    def fn() -> None:
        return None, {"return_direct": 'image' in name}
    return StructuredTool.from_function(func=fn,name=name,description=name,response_format='content_and_artifact')


boss_ai_name = 'primary'
sub_ai_names = [
    'search',
    'image',
    ]

# define tools

ai_names = [boss_ai_name] + sub_ai_names
ai_tools = defaultdict(list)

# add the tool or switching between tasks
def _create_switch_ai_tool(ai_name):
    return StructuredTool.from_function(
        func=lambda: (ai_name, {"to_ai":ai_name}), 
        response_format='content_and_artifact', 
        name=f"switch_to_{ai_name}_ai",
        description=f"switch to {ai_name} ai")
to_sub_ai_tools = [_create_switch_ai_tool(sub_ai_name) for sub_ai_name in sub_ai_names]
to_boss_ai_tool = [_create_switch_ai_tool(boss_ai_name)]
to_ai_tool_names:Set[str] = {tool.name for tool in to_sub_ai_tools + to_boss_ai_tool}
ai_tools[boss_ai_name].extend(to_sub_ai_tools)
for sub_ai_name in sub_ai_names:
    ai_tools[sub_ai_name].extend(to_boss_ai_tool)

# create fake tools
for ai_name in [boss_ai_name] + sub_ai_names:
    ai_tools[ai_name].extend([_fake_tool(f'{ai_name}_tool_{i}') for i in range(2)])

# mock tools
@tool
def get_user_info(config: RunnableConfig):
    """get_user_info"""
    configuration = config.get("configurable", {})
    logging.info("get_user_info")
    return {"email":"abc.com"}

# define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: NotRequired[str]
    current_ai: NotRequired[str]

# create fake assistant
class AINode:
    def __init__(self, name:str):
        self.name = name
    def __call__(self, state:State) -> State:
        use_tool = random.choice([True, False])
        if use_tool and ai_tools[self.name]:
            tool:BaseTool = random.choice(ai_tools[self.name])
            tool_calls=[{'name': tool.name, 'args': {}, 'id': f'call_{uuid4()}', 'type': 'tool_call'}]
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            message = AIMessage(content=f'I am {self.name}')
        return {"messages":[message]}

# custom nodes

def initialise_node(state:State):
    updates:State = {'messages':[]}
    if 'current_ai' not in state:
        updates['current_ai'] = boss_ai_name
    if 'user_info' not in state:
        updates['user_info'] = get_user_info.invoke({})
    return updates

def route_to_ai(state:State):
    # TODO: support parallel toolcalls yet
    updates:State = {'messages':[]}
    message = state['messages'][-1]
    # route based on previous toolcall
    if isinstance(message, ToolMessage) and  message.name in to_ai_tool_names:
        updates['current_ai'] = message.artifact['to_ai']
    return updates

# custom edges
to_current_ai_kwargs = {
    'path':lambda state:state['current_ai'],
    'path_map':{k:k for k in ai_names}
    }
def route_to_ai_or_end(state:State) -> Literal['route_to_ai', '__end__']:
    message = state['messages'][-1]
    assert isinstance(message, ToolMessage)
    if isinstance(message.artifact, dict) and message.artifact.get('return_direct'):
        return "__end__"
    return 'route_to_ai'


def get_builder():
    builder = StateGraph(State)
    builder.add_edge(START, "initialize")
    builder.add_node("initialize", initialise_node)
    builder.add_edge("initialize", "route_to_ai")
    builder.add_node("route_to_ai", route_to_ai)
    builder.add_conditional_edges("route_to_ai", lambda state:state['current_ai'], {k:k for k in ai_names})
    # assistants
    for ai_name in ai_names:
        tools_node_name = f"{ai_name}_tools"
        builder.add_node(ai_name, AINode(ai_name))
        if ai_tools[ai_name]:
            builder.add_conditional_edges(ai_name, tools_condition, {"tools":tools_node_name, END:END})
            builder.add_node(tools_node_name, ToolNode(ai_tools[ai_name]))
            builder.add_conditional_edges(tools_node_name, route_to_ai_or_end)
        else:
            builder.add_edge(ai_name, END)

    return builder

TOOL_NAMES = set()
for tools in ai_tools.values():
    for t in  tools:
       TOOL_NAMES.add(t.name)

if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.INFO)
    memory = MemorySaver()
    graph = get_builder().compile(
        checkpointer=memory, 
        interrupt_before=[f'{name}_tools' for name in ai_names],
        )
    fname, _ = os.path.splitext(os.path.basename(__file__))
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path=f'graphs/{fname}.png')
    config = {"configurable":{"thread_id":str(uuid4())}}
    msgs = [
        HumanMessage(content="Tell me about qingyi"),
        # HumanMessage(content="用中文"),
        # HumanMessage(content="写一段关于她的小说吧。"),
    ]
    try:
        for _ in range(10): 
            if graph.get_state(config).next:
                # approve = input("approve:")
                inputs = None
            else:
                msg = f"human: {msgs.pop(0)}" if msgs else input("human: ")
                inputs = {"messages":[msg]}
            events = graph.stream(inputs, config, stream_mode='updates')
            for updates in events:
                for node, updates in updates.items():
                    print(f"== {node} ==")
                    for k,v in updates.items():
                        if k == 'messages':
                            messages = updates.get('messages', [])
                            if not isinstance(messages, list):
                                messages = [messages]
                            for message in messages:
                                print(f"{message.type}: {message}")
                                print()
                        else:
                            print(f"{k}: {v}")
                    print()
    finally:
        print(graph.get_state(config))