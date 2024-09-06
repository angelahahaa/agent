import base64
import tempfile
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (ConfigurableField, Runnable,
                                      RunnableConfig, RunnableLambda, RunnablePassthrough)
from langchain_core.tools import tool
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


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "search",
            ]
        ],
        update_dialog_stack,
    ]

@tool
def spell_backwards(word:str, config: RunnableConfig) -> str:
    """ Spells the word backwards.
    """
    return word[::-1]

@tool
def to_primary_assistant(
        reason:Annotated[bool, "reason for completion or escalation."]
    ):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the primary assistant,
    who can re-route the dialog based on the user's needs."""
    return 

@tool
def to_search_assistant(
    query:Annotated[str, "search query to look up"],
    ):
    """ Transfers work to a specialized assistant to handle searches. 
    The search assistant has knowledge on latest events and user uploaded contents.
    """
    return

specialized_assistant_tools = [to_search_assistant]
specialized_assistant_tool_names = {t.name for t in specialized_assistant_tools}
primary_tools = [spell_backwards]
class PrimaryAssistant:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = primary_tools + specialized_assistant_tools

    def __call__(self, state, config:RunnableConfig):
        runnable = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant for Virtuos Games Company. "
                        " Reply using your own knowledge. Use the provided tools to assist the user when you lack knowledge. "
                        "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                    ),
                    ("placeholder", "{messages}"),
                ]
            )
            | self.llm.with_config(config).bind_tools(self.tools) 
            )
        if "user_info" not in state:
            state["user_info"] = ""
        response = runnable.invoke(state)
        # response = await runnable.ainvoke(state)
        return {"messages": response}

def create_entry_node(new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        if new_dialog_state == 'primary_assistant':
            content=f"Resuming dialog with {new_dialog_state}. Please reflect on the past conversation and assist the user as needed."
        else:
            content=f"The assistant is now the {new_dialog_state}. Reflect on the above conversation between the host assistant and the user."\
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {new_dialog_state},"\
                    " your task is not complete until after you have successfully invoked the appropriate tool."\
                    " If the user changes their mind or needs help for other tasks, call the to_primary_assistant function to let the primary host assistant take control."\
                    " Do not mention who you are - just act as the proxy for the assistant."
        return {
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node

def route_to_workflow(
        state: State,
    ) -> Literal[
        "primary_assistant",
        "search_assistant",
    ]:
    dialog_state = state.get("dialog_state") or ["primary_assistant"]
    return dialog_state[-1]

def route_primary_assistant(
    state: State,
    ) -> Literal[
        "primary_tools",
        "to_search_assistant",
        "__end__",
    ]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        name = tool_calls[0]["name"]
        if name in specialized_assistant_tool_names:
            return name
    return "primary_tools"

def route_specialized_assistant(
        state: State,
    ) -> Literal[
        "tools",
        "to_primary_assistant",
        "__end__",
    ]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        name = tool_calls[0]["name"]
        if name == "to_primary_assistant":
            return f"to_primary_assistant"
    return "tools"

def get_graph():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    builder = StateGraph(State)

    builder.add_edge(START, "init")
    builder.add_node("init", lambda state:{"user_info":""})
    builder.add_conditional_edges("init", route_to_workflow)
    # primary assistant
    AI = "primary_assistant"
    TOOLS = f"primary_tools"
    assistant = PrimaryAssistant(llm)
    builder.add_node(f"to_{AI}", create_entry_node(AI))
    builder.add_edge(f"to_{AI}", AI)
    builder.add_node(AI, assistant)
    builder.add_conditional_edges(AI, route_primary_assistant)
    builder.add_node(TOOLS, ToolNode(assistant.tools))
    builder.add_edge(TOOLS, AI)
    # search assistant
    AI = "search_assistant"
    TOOLS = f"search_tools"
    assistant = SearchAssistant(llm)
    assistant.tools += [to_primary_assistant]
    builder.add_node(f"to_{AI}", create_entry_node(AI))
    builder.add_edge(f"to_{AI}", AI)
    builder.add_node(AI, assistant)
    builder.add_conditional_edges(AI, route_specialized_assistant, path_map={END:END,'tools':TOOLS,'to_primary_assistant':'to_primary_assistant'})
    builder.add_node(TOOLS, ToolNode(assistant.tools))
    builder.add_edge(TOOLS, AI)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path=f'graphs/primary_assistant.png')
    return graph
if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv('.env')
    from uuid import uuid4
    graph = get_graph()
    config = {"configurable":{"thread_id":str(uuid4())}}
    msgs = [
        HumanMessage(content="Tell me about qingyi"),
        HumanMessage(content="用中文"),
        HumanMessage(content="写一段关于她的小说吧。"),
    ]
    while True: 
        if msgs:
            msg = msgs.pop(0)
        else:
            msg = input("human:")
        if not msg:
            break
        inputs = {"messages":[msg]}
        events = graph.stream(inputs, config, stream_mode="updates")
        for event in events:
            for k, v in event.items():
                print(f"== {k} ==")
                print()
                messages = v.get('messages', [])
                if not isinstance(messages, list):
                    messages = [messages]
                for message in messages:
                    print(f"{message.type}:{message}")
                    print()


