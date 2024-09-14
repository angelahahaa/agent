
from typing import (Annotated, List, Literal, TypedDict)

from langgraph.graph import END
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

from chatbot.architecture.multiagent import State

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    
def return_direct_condition(state:State) -> Literal['__end__','agent']:
    message = state['messages'][-1]
    if isinstance(message.artifact, dict) and message.artifact.get('return_direct'):
        return END
    return 'agent'