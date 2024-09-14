import dotenv

dotenv.load_dotenv('.env')
from typing import (Annotated, Dict, List, Type, TypedDict, get_type_hints)

from langchain_core.runnables import (Runnable,
                                      RunnableConfig, RunnableLambda)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.tool_node import ToolNode
from typing_extensions import TypedDict



class ReactAgentState(TypedDict):
    messages:Annotated[List[AnyMessage], add_messages]

def react_agent_builder(
    runnable:Runnable, 
    tools:List[BaseTool], 
    state_schema:Type[Dict]=ReactAgentState,
) -> StateGraph:
    assert  'messages' in get_type_hints(state_schema), "Key 'messages' is missing from State definition"
    def invoke(state, config: RunnableConfig):
        return {"messages":runnable.invoke(state, config)}
    async def ainvoke(state, config: RunnableConfig):
        return {"messages":await runnable.ainvoke(state, config)}
    agent = RunnableLambda(func=invoke, afunc=ainvoke)
    builder = StateGraph(state_schema)
    builder.add_edge(START, 'agent')
    builder.add_node('agent', agent)
    builder.add_conditional_edges('agent', tools_condition)
    builder.add_node('tools', ToolNode(tools))
    builder.add_edge('tools','agent')
    return builder