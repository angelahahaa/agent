
from datetime import datetime
from typing import Annotated, Any, List, Literal, Tuple, TypedDict, Union, cast

from langchain_core.messages import (AIMessage, AnyMessage, ToolCall,
                                     ToolMessage)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from chatbot.architecture._multiagent import State
from langchain_core.runnables import (RunnableLambda, RunnablePassthrough)

from langchain_core.language_models.chat_models import BaseChatModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableLambda, RunnablePassthrough)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def no_tools_agent_builder(llm: BaseChatModel):
    """ agent with no tools, only default system prompt"""
    builder = StateGraph(State)
    agent = (RunnablePassthrough.assign(**{"time": lambda x: datetime.now()}) | ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for Virtuos Games Company. "
         "\nCurrent time: {time}."),
        ("placeholder", "{messages}"),
    ]) | llm | RunnableLambda(lambda x: {'messages': [x]}))
    builder.add_edge(START, 'agent')
    builder.add_node('agent', agent)
    builder.add_edge('agent', END)
    return builder
    
def return_direct_condition(state:State) -> Literal['__end__','agent']:
    for message in state['messages'][::-1]:
        if isinstance(message, ToolMessage) and \
            isinstance(message.artifact, dict) and \
                message.artifact.get('return_direct'):
            continue
        return 'agent'
    return END

class PartialToolNode(ToolNode):
    """Similar to langgraph.prebuilt.ToolNode, but instead of looking at messages[-1], it will look at the latest AI message. 
    Any ToolMessage after the latest AIMessage will be assumed as processed and those tools will not be run.

    The `PartialToolNode` is roughly analogous to:

    ```python
    tools_by_name = {tool.name: tool for tool in tools}
    def tool_node(state: dict):
        result = []
        processed_tool_call_ids = set()
        for message in state["messages"][::-1]:
            if isinstance(message, AIMessage):
                break
            elif isinstance(message, ToolMessage):
                processed_tool_call_ids.add(message.tool_call_id)
            else:
                raise 
        for tool_call in message.tool_calls:
            if tool_call["id"] in processed_tool_call_ids:
                continue 
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}    
    ```

    Important:
        - The state MUST contain a list of messages.
        - There MUST be an `AIMessage`.
        - If there are messages after the latest `AIMessage`, they must be `ToolMessage`.

    """
    def _parse_input(
        self, input: Union[list[AnyMessage], dict[str, Any]]
    ) -> Tuple[List[ToolCall], Literal["list", "dict"]]:
        if isinstance(input, list):
            output_type = "list"
            messages = input
        elif messages := input.get("messages", []):
            output_type = "dict"
        else:
            raise ValueError("No message found in input")
        
        processed_tool_call_ids = set()

        for message in reversed(messages):
            if isinstance(message, AIMessage):
                break
            elif isinstance(message, ToolMessage):
                processed_tool_call_ids.add(message.tool_call_id)
            else:
                raise TypeError(f"Unexpected message type: {type(message).__name__}")
        
        if not isinstance(message, AIMessage):
            raise ValueError("No AIMessage found")

        tool_calls = [
            self._inject_state(call, input)
            for call in cast(AIMessage, message).tool_calls 
            if call['id'] not in processed_tool_call_ids
        ]
        return tool_calls, output_type