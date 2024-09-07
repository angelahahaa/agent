import dotenv
dotenv.load_dotenv('.env')

import logging
from typing import Annotated, NotRequired
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

logger = logging.getLogger()
from chatbot.agent.multiagent import (AINode, State, create_multiagent_graph)

from datetime import datetime
from typing import Annotated

from langchain_core.messages import (HumanMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (ConfigurableField, RunnableConfig, RunnablePassthrough)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI
from typing import Annotated
from langchain_core.messages import (HumanMessage)

from chatbot import tools
def initialise_node(state:State):
    updates:State = {'messages':[]}
    if 'user_info' not in state:
        updates['user_info'] = tools.get_user_info.invoke({})
    return updates

class CustomState(State):
    user_info: NotRequired[str]


class PrimaryAI(AINode):
    def __init__(self, llm):
        self.llm = llm
        super().__init__('primary', [])
    def __call__(self, state: State, config: RunnableConfig) -> State:
        runnable = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant for Virtuos Games Company. "
                        " Reply using your own knowledge first and only use the provided tools to assist the user when you lack knowledge. "
                        "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                    ),
                    ("placeholder", "{messages}"),
                ]
            )
            | self.llm.with_config(config).bind_tools(self.tools) 
            )
        return {"messages": runnable.invoke(state)}
    def switch_ai_tool(self):
        @tool(response_format='content_and_artifact')
        def switch_to_primary_ai(
                reason:Annotated[bool, "reason for completion or escalation."],
            ):
            """A tool to mark the current task as completed and/or to escalate control of the dialog to the primary assistant,
            who can re-route the dialog to an search, image or IT agent based on the user's needs."""
            return (
                f"Resuming dialog with '{self.name}'. Please reflect on the past conversation and assist the user as needed.", 
                {"to_ai":self.name}
                )
        return switch_to_primary_ai
    

class SearchAI(AINode):
    def __init__(self, llm):
        self.llm = llm
        super().__init__('search', [
            tools.search_internet, 
            tools.search_session_data,
            ])
    def __call__(self, state: State, config: RunnableConfig) -> State:
        runnable = (
            RunnablePassthrough.assign(**{'session_data_summary':tools.get_session_data_summary})
            | ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant for Virtuos Games Company. "
                        " Use the provided tools to search for information to assist the user's queries. "
                        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                        " If a search comes up empty, expand your search before giving up."
                        f" You can use the tools to refer the task to other assistants if the user is no longer requiring {self.name}. "
                        "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                        "\n\nCurrent Session Data Summary:\n<User>\n{session_data_summary}\n</User>"
                        "\nCurrent time: {time}.",
                    ),
                    ("placeholder", "{messages}"),
                ]
            ).partial(time=datetime.now())
            | self.llm.with_config(config).bind_tools(self.tools) 
            )
        return {"messages": runnable.invoke(state)}
    def switch_ai_tool(self):
        @tool(response_format='content_and_artifact')
        def switch_to_search_ai(
                query:Annotated[str, "search query to look up"],
            ):
            """ Transfers work to a specialized assistant to handle searches. 
            The search assistant has knowledge on latest events and user uploaded contents.
            """
            content=f"The assistant is now the '{self.name}'. Reflect on the above conversation between the host assistant and the user."\
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {self.name},"\
                    " your task is not complete until after you have successfully invoked the appropriate tool."\
                    f" You can use the tools to refer the task to other assistants if the user is no longer requiring {self.name}. "\
                    " Do not mention who you are - just act as the proxy for the assistant."
            return (
                content, 
                {"to_ai":self.name}
                )
        return switch_to_search_ai
    
class ImageAI(AINode):
    def __init__(self, llm):
        self.llm = llm
        super().__init__('image', [
            tools.generate_image_with_text,
            ])
    def __call__(self, state: State, config: RunnableConfig) -> State:
        runnable = (
            RunnablePassthrough.assign(**{'session_data_summary':tools.get_session_data_summary})
            | ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant for Virtuos Games Company. "
                        " Use the provided tools to generate images. "
                        "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                        "\nCurrent time: {time}.",
                    ),
                    ("placeholder", "{messages}"),
                ]
            ).partial(time=datetime.now())
            | self.llm.with_config(config).bind_tools(self.tools) 
            )
        return {"messages": runnable.invoke(state)}
    def switch_ai_tool(self):
        @tool(response_format='content_and_artifact')
        def switch_to_image_ai(
                reason:Annotated[bool, "reason to switch to image assistant."],
            ):
            """ Transfers work to a specialized assistant to handle image. 
            The image assistant has the ability to send created images to the user.
            """
            content=f"The assistant is now the '{self.name}'. Reflect on the above conversation between the host assistant and the user."\
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {self.name},"\
                    " your task is not complete until after you have successfully invoked the appropriate tool."\
                    f" You can use the tools to refer the task to other assistants if the user is no longer requiring {self.name}. "\
                    " Do not mention who you are - just act as the proxy for the assistant."
            return (
                content, 
                {"to_ai":self.name}
                )
        return switch_to_image_ai

import dotenv
dotenv.load_dotenv('.env')
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).configurable_fields(
        model_name=ConfigurableField(
                id="llm/model_name",
                is_shared=True,
            ),
        temperature=ConfigurableField(
                id="llm/temperature",
                is_shared=True,
            ),
        )
boss_ai_nodes = [
    # ImageAI(llm),
    # PrimaryAI(llm),
    ]
sub_ai_nodes = [
    PrimaryAI(llm),
    SearchAI(llm),
    ImageAI(llm),
]
tool_node_names = [f'{node.name}_tools' for node in boss_ai_nodes + sub_ai_nodes]
tool_names = set()
for node in boss_ai_nodes + sub_ai_nodes:
    for t in  node.tools:
        tool_names.add(t.name)


builder = StateGraph(CustomState)
builder.add_edge(START, "initialize")
builder.add_node("initialize", initialise_node)
graph = create_multiagent_graph(boss_ai_nodes, sub_ai_nodes, start='initialize', builder=builder)
memory = MemorySaver()
graph = graph.compile(
    checkpointer=memory, 
    interrupt_before=tool_node_names,
    )
for node in sub_ai_nodes:
    print(f"==== {node.name} ====")
    for t in node.tools:
        print(t)
        print()
    print("===")

import os
fname, _ = os.path.splitext(os.path.basename(__file__))
graph.get_graph(xray=True).draw_mermaid_png(output_file_path=f'graphs/{fname}.png')

if __name__ == '__main__':

    config = {"configurable":{"thread_id":str(uuid4())}}
    msgs = [
        HumanMessage(content="Tell me about qingyi"),
        # HumanMessage(content="用中文"),
        # HumanMessage(content="写一段关于她的小说吧。"),
    ]
    try:
        for _ in range(10): 
            if graph.get_state(config).next:
                approve = input("approve:")
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