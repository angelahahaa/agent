import logging
import random
from typing import Annotated, List, Literal, NotRequired, Set, TypedDict
from uuid import uuid4

from langchain.tools.base import StructuredTool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

logger = logging.getLogger()

def _fake_tool(name:str, return_direct:bool=False) -> BaseTool:
    if return_direct:
        def fn() -> None:
            return None, {"return_direct": 'image' in name}
        return StructuredTool.from_function(func=fn,name=name,description=name,response_format='content_and_artifact')
    else:
        def fn() -> None:
            return 
        return StructuredTool.from_function(func=fn,name=name,description=name,response_format='content')
        
def create_switch_ai_tool(ai_name, role:Literal['primary', 'specialist','foo'] = 'foo', additional_description=""):
    """ creates a simple tool for switching to a new ai, please customise for better results"""
    if role == 'primary':
        content=f"Resuming dialog with {ai_name}. Please reflect on the past conversation and assist the user as needed."
    elif role == 'specialist':
        content = f"The assistant is now the {ai_name}. Reflect on the above conversation between the host assistant and the user."\
                f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {ai_name},"\
                " your task is not complete until after you have successfully invoked the appropriate tool."\
                " If the user changes their mind or needs help for other tasks, call the to_primary_assistant function to let the primary host assistant take control."\
                " Do not mention who you are - just act as the proxy for the assistant."
    else:
        content = ai_name # do not use this, just for easy testing
    return StructuredTool.from_function(
        func=lambda: (content, {"to_ai":ai_name}), # {"to_ai":ai_name} is IMPORTANT!
        response_format='content_and_artifact', 
        name=f"switch_to_{ai_name}_ai",
        description=f"Transfers work to a specialized assistant {ai_name}. " + additional_description)

# define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    current_ai: NotRequired[str]

# define AI node
class AINode:
    def __init__(self, name:str, tools:List):
        self.name = name
        self.tools = tools
    def __call__(self, state:State, config:RunnableConfig) -> State:
        """ This one is fake, it just returns a random tool from the list or a message 
        """
        use_tool = random.choice([True, False])
        if use_tool and self.tools:
            tool:BaseTool = random.choice(self.tools)
            tool_calls=[{'name': tool.name, 'args': {}, 'id': f'call_{uuid4()}', 'type': 'tool_call'}]
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            message = AIMessage(content=f'I am {self.name}')
        return {"messages":[message]}
    def switch_ai_tool(self):
        """ Tool switching back to this AI Node. Has to return artifact {"to_ai":ai_name}
        """
        return create_switch_ai_tool(self.name, 'foo')
    def __repr__(self) -> str:
        return f"AINode(name={self.name}, tools={self.tools})"



def create_multiagent_graph(
        boss_ai_nodes:List[AINode], 
        sub_ai_nodes:List[AINode], 
        start=START, end=END, 
        builder:StateGraph|None=None,
        ) -> StateGraph:
    """ if there is no boss node, all agents can pass work to each other, default agent will be the first one in the list.
    """
    boss_ai_names = [node.name for node in boss_ai_nodes]
    sub_ai_names = [node.name for node in sub_ai_nodes]

    ai_nodes = boss_ai_nodes + sub_ai_nodes
    assert len(ai_nodes) > 0, "Need at least one AI."
    ai_names = boss_ai_names + sub_ai_names

    # add the tool for switching between tasks
    if boss_ai_nodes:
        to_sub_ai_tools = [node.switch_ai_tool() for node in sub_ai_nodes]
        to_boss_ai_tools = [node.switch_ai_tool() for node in boss_ai_nodes]

        _ = [node.tools.extend(to_sub_ai_tools) for node in boss_ai_nodes]
        _ = [node.tools.extend(to_boss_ai_tools) for node in sub_ai_nodes]
    else:
        # they can all talk to each other, yay!
        for node in sub_ai_nodes:
            node.tools.extend([n.switch_ai_tool() for n in sub_ai_nodes if n!=node])
    to_ai_tool_names:Set[str] = {node.switch_ai_tool().name for node in ai_nodes}

    # custom nodes
    def route_to_ai(state:State):
        """ Updates current_ai depending on previous message
        """
        # TODO: support parallel toolcalls yet
        updates:State = {'messages':[]}
        message = state['messages'][-1]
        # no ai set yet, go to primary
        if 'current_ai' not in state:
            updates['current_ai'] = boss_ai_names[0] if boss_ai_names else sub_ai_names[0]
        # previous tool call requires a switch, do it
        if isinstance(message, ToolMessage) and  message.name in to_ai_tool_names:
            updates['current_ai'] = message.artifact['to_ai']
        return updates

    # custom edges
    def route_to_ai_or_end(state:State) -> Literal['route_to_ai', '__end__']:
        """ Decides on where to go after tool calls
        END if tool create artifact dict {"return_direct":True}"""
        message = state['messages'][-1]
        assert isinstance(message, ToolMessage)
        if isinstance(message.artifact, dict) and message.artifact.get('return_direct'):
            return "__end__"
        return 'route_to_ai'


    builder = builder or StateGraph(State)
    builder.add_edge(start, "route_to_ai")
    builder.add_node("route_to_ai", route_to_ai)
    builder.add_conditional_edges("route_to_ai", lambda state:state['current_ai'], {k:k for k in ai_names})
    # assistants
    for ai_node in ai_nodes:
        tools_node_name = f"{ai_node.name}_tools"
        builder.add_node(ai_node.name, ai_node)
        if ai_node.tools:
            builder.add_conditional_edges(ai_node.name, tools_condition, {"tools":tools_node_name, END:end})
            builder.add_node(tools_node_name, ToolNode(ai_node.tools))
            builder.add_conditional_edges(tools_node_name, route_to_ai_or_end, {END:end, 'route_to_ai':'route_to_ai'})
        else:
            builder.add_edge(ai_node.tools, end)

    return builder