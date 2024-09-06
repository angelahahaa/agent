from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableConfig, RunnablePassthrough, RunnableLambda)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from typing import TypedDict
@tool
def search_session_data(query:str):
    """A search engine optimized for comprehensive, accurate, and trusted results. 
    Useful for when you need to answer questions about data or files uploaded in current session.
    Input should be a search query.

    Args:
        query: search query to look up

    Returns:
        A list of dictionaries, each containing the 'source' of the data and the 'content' that matches the query.
    """
    return [
        {"source":"zzz_characters.csv","content":"Name:Jane Doe\nHair colour:Black\nDMG:Physical"},
        {"source":"zzz_characters.csv","content":"Name:Qingyi\nHair colour:Green\nDMG:Electrical"},
    ]

# search_internet = TavilySearchResults(max_results=1, name='search_internet')
@tool
def search_internet(query:str):
    """A search engine optimized for comprehensive, accurate, and trusted results. 
    Useful for when you need to answer questions about current events. 
    Input should be a search query.
    
    Args:
        query: search query to look up
    """
    ...

def _get_session_data_summary(config:RunnableConfig) -> str:
    """ fetch a string that describes what users have in their session file database.
    """
    return """
    zzz_characters.csv: data for Jane Doe (Black hair, Physical DMG), Qingyi (Green hair, Electro DMG)
    Character Design.pptx: details Elara Stormwind, a mage with elemental spirit summoning ability, questing against a dark prophecy.
    """


tools = [
    search_internet,
    search_session_data,
]


class SearchAssistant:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = tools

    def __call__(self, state, config:RunnableConfig):
        runnable = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant for Virtuos Games Company. "
                        " Use the provided tools to search for information to assist the user's queries when asked or when you lack knowledge. "
                        " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                        " If a search comes up empty, expand your search before giving up."
                        "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                        "\n\nCurrent Session Data Summary:\n<User>\n{session_data_summary}\n</User>"
                        "\nCurrent time: {time}.",
                    ),
                    ("placeholder", "{messages}"),
                ]
            ).partial(time=datetime.now(), session_data_summary=_get_session_data_summary(config))
            | self.llm.with_config(config).bind_tools(self.tools) 
            )
        if "user_info" not in state:
            state["user_info"] = ""
        response = runnable.invoke(state)
        # response = await runnable.ainvoke(state)
        return {"messages": response}
    
async def main():
    import dotenv
    dotenv.load_dotenv('.env')
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import tools_condition
    from langchain_openai import ChatOpenAI
    from langgraph.graph.message import AnyMessage, add_messages
    from typing import Annotated
    from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage, ToolMessage)
    from uuid import uuid4
    def add_react(builder:StateGraph, assistant, prefix="", start=START, end=END):
        AI = f"{prefix}assistant"
        TOOLS = f"{prefix}tools"
        builder.add_edge(START, AI)
        builder.add_node(AI, assistant)
        builder.add_conditional_edges(AI, tools_condition, path_map={END:end, 'tools':TOOLS})
        builder.add_node(TOOLS, ToolNode(assistant.tools))
        builder.add_edge(TOOLS, AI)
        return builder
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        user_info: str
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prefix = 'search_'
    graph = add_react(StateGraph(State), SearchAssistant(llm), prefix=prefix).compile(checkpointer=MemorySaver())
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path=f'graphs/{prefix}assistant.png')

    inputs = {"messages":[HumanMessage(content="Tell me about qingyi")]}
    config = {"configurable":{"thread_id":str(uuid4())}}
    async for event in graph.astream_events(inputs, config, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")
if __name__ == '__main__':
    import asyncio
    asyncio.run(main())