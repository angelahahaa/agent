import logging
from functools import partial
from typing import (Annotated, Any, Dict, List, Literal, Set, Type, TypedDict,
                    TypeVar, get_type_hints)

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (Runnable, RunnableConfig, RunnableLambda,
                                      RunnablePassthrough)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, AfterValidator, Field, model_validator, SkipValidation
from typing_extensions import TypedDict
from chatbot.mocks import MockChat, mock_tool
from chatbot.architecture.base import return_direct_condition
from langchain_core.language_models.chat_models import BaseChatModel
logger = logging.getLogger(__name__)

POP='<pop>'
PopType = Literal['<pop>']

def extend_or_pop(left:List[str], right:List[str]):
    if not right:
        return left
    if POP in right:
        left = left[:-1]
    return left + [r for r in right if r != POP]

class MultiAgentState(TypedDict):
    messages:Annotated[AnyMessage, add_messages]
    current_agent:Annotated[List[str], extend_or_pop]
    pending_message:AIMessage

def assert_has_bind_tools(agent):
    assert hasattr(agent, "bind_tools"), "agent.bind_tools not implemented."

class Agent(BaseModel):
    name:str
    prompt:SkipValidation[ChatPromptTemplate]=Field(default=None)
    llm:SkipValidation[BaseChatModel]
    tools:List[SkipValidation[BaseTool]]=Field(default_factory=list)
    enter_tool:SkipValidation[BaseTool]=Field(default='default')
    exit_tool:SkipValidation[BaseTool]|None=Field(default='default')

    @model_validator(mode='after')
    def after_validator(self):
        assert hasattr(self.llm, "bind_tools"), "llm.bind_tools not implemented."
        if self.prompt is None:
            self.prompt = ChatPromptTemplate.from_messages([("placeholder", "{messages}")])
        if self.enter_tool == 'default':
            logger.warning(f".enter_tool not defined for agent '{self.name}', using default. It is strongly recommended to define a custom as_tool for each agent instead.")
            self.enter_tool = StructuredTool.from_function(
                func=lambda:f"The assistant is now specialised agent '{self.name}'. Please reflect on the past conversation and assist the user as needed.",
                description=f"Transfers work to {self.name}",
                name=f"to_{self.name}"
                )
        if self.exit_tool == 'default':
            logger.warning(f".exit_tool not defined for agent '{self.name}', using default. It is strongly recommended to define a custom as_tool for each agent instead.")
            self.exit_tool = StructuredTool.from_function(
                func=lambda :f"Exiting dialog with specialised agent '{self.name}'. Please reflect on the past conversation and assist the user as needed.",
                description=f"Marks task of {self.name} complete or canceled.",
                name=f"exit_{self.name}"
                )
        return self

    def to_node(self):
        tools = self.tools
        if self.exit_tool:
            tools.append(self.exit_tool)
        llm = (self.prompt | self.llm.bind_tools(tools))
        def invoke(state, config: RunnableConfig):
            return {
                "pending_message": llm.invoke(state, config),
                }
        async def ainvoke(state, config: RunnableConfig):
            return {
                "pending_message": await llm.ainvoke(state, config),
                }
        return RunnableLambda(invoke, ainvoke, self.name)

def process_pending_message(state:MultiAgentState, config:RunnableConfig, enter_tools:dict, exit_tools:set):
    updates = {'pending_message':None}
    pending_message = state['pending_message']
    if not pending_message.tool_calls or not any(tc['name'] in enter_tools for tc in pending_message.tool_calls):
        updates['messages'] = [pending_message]
    else:
        # valid_tool_calls = []
        updates['current_agent'] = []
        for tc in pending_message.tool_calls:
            if tc['name'] in enter_tools:
                updates['current_agent'].append(enter_tools[tc['name']])
            elif tc['name'] in exit_tools:
                updates['current_agent'].append(POP)
            # else:
            #     valid_tool_calls.append(tc)
        # if valid_tool_calls:
        #     pending_message.tool_calls = valid_tool_calls
    updates['messages'] = [pending_message]

    return updates


def multi_agent_builder(
    agents:List[Agent], # order matters, if there is no current ai, always routes to the first one
    *,
    builder:StateGraph | None = None,
    start:str=START,
    end:str=END,
) -> StateGraph:
    # assert stuff
    assert len(agents) > 1, "Are you dumb? don't use multiagent if you have don't have multiple agents."
    all_agent_names = [a.name for a in agents]
    assert len(all_agent_names) == len(set(all_agent_names)), "All agents should have different names."
    state_schema = builder.schema if builder else MultiAgentState
    assert  'messages' in get_type_hints(state_schema), "Key 'messages' is missing from state_schema definition"
    assert  'pending_message' in get_type_hints(state_schema), "Key 'pending_messages' is missing from state_schema definition"
    assert  'current_agent' in get_type_hints(state_schema), "Key 'current_agent' is missing from state_schema definition"

    # define tools node
    added_tool_names = set()
    all_tools = []
    for agent in agents:
        for tool in agent.tools + [agent.enter_tool, agent.exit_tool]:
            if tool and tool.name not in added_tool_names:
                added_tool_names.add(tool.name)
                all_tools.append(tool)
    tools_node = ToolNode(all_tools)

    # nodes
    ppm_node = RunnableLambda(
        func = partial(process_pending_message, 
                       enter_tools={agent.enter_tool.name:agent.name for agent in agents},
                       exit_tools={agent.exit_tool.name for agent in agents if agent.exit_tool},
                       ), 
        name = 'process_pending_message')
    def select_agent(state:MultiAgentState):
        return {'current_agent':[] if state.get('current_agent') else [all_agent_names[0]]}


    # routes
    def route_process_pending_message(state:MultiAgentState):
        if not state['messages'] or not isinstance(state['messages'][-1], AIMessage):
            return 'select_agent'
        return tools_condition(state)

    # build
    builder = builder or StateGraph(state_schema)
    builder.add_edge(start, 'select_agent')
    builder.add_node('select_agent', select_agent)
    builder.add_conditional_edges('select_agent', lambda s:s['current_agent'][-1], all_agent_names)
    for agent in agents:
        builder.add_node(agent.name, agent.to_node())
        builder.add_edge(agent.name, ppm_node.name)
    builder.add_node(ppm_node.name, ppm_node)
    builder.add_conditional_edges(ppm_node.name, route_process_pending_message, {END:end, "tools":tools_node.name, 'select_agent':'select_agent'})
    builder.add_node('tools', tools_node)
    builder.add_edge(tools_node.name, 'select_agent')
    return builder
def save_graph():
    import os
    agents = [Agent(
    name=name,
    llm=MockChat(model=name),
    tools=[mock_tool(f'{name}1'), mock_tool(f'{name}2')],
    ) for name in ['primary','worker1','worker2']]

    graph = multi_agent_builder(agents).compile()

    graph.get_graph(
    # xray=True,
    ).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')
# save_graph()