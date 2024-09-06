import dotenv

dotenv.load_dotenv('.env')
import base64
import tempfile
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     RemoveMessage, SystemMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (ConfigurableField, Runnable,
                                      RunnableConfig, RunnableLambda)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from chatbot.database import vector_db

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
# tools
tavily_search = TavilySearchResults(max_results=1, name='web_search')

@tool
def generate_text_image(text:str, size=(256,256)):
    """ returns base64 encoded image generation given a prompt. Default size is 256 x 256
    """
    from PIL import Image, ImageDraw, ImageFont
    import io
    import base64

        # Create a new image with white background
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    
    # Define the text and font (defaulting to a built-in PIL font)
    font = ImageFont.load_default()
    
    # Calculate the bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=font)
    
    # Calculate position to center the text
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2
    
    # Draw the text on the image
    draw.text((x, y), text, font=font, fill="black")

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Format the response
    # message = AIMessage(content=[{
    #     "type": "image_url",
    #     "image_url": {
    #         "url": f"data:image/png;base64,{encoded_image}",
    #         "detail": "low",
    #     },
    # }])
    return {'image':encoded_image}

@tool
def user_database_metadata(config:RunnableConfig) -> Dict:
    """ Returns the metadata of a users database as dictionary, including the data source. 
    """
    configuration = config.get("configurable", {})
    collection_name = configuration['thread_id']
    return vector_db.get_database_info(collection_name)

@tool
def spell_backwards(word:str, config: RunnableConfig) -> str:
    """ Spells the word backwards.
    """
    return word[::-1]

@tool
def get_user_info(config: RunnableConfig) -> dict:
    """ Fetch all user information.
    Returns:
        A dictionary of user information. Returns an empty dictionary if no user information found.
    """
    configuration = config.get("configurable", {})
    email = configuration.get("email", "")
    info = {}
    if ('angela' in email) or ('shiya' in email):
        info = {'name':'Peng Shiya', 'department':'RnD', 'studio':'SHA', 'position':'Engineer', 'location':'Shanghai'}
    elif 'pengseng' in email:
        info =  {'name':'Ang Peng Seng', 'department':'RnD', 'studio':'SGP', 'position':'Lead Engineer', 'location':'Singapore'}
    elif 'yuyong' in email:
        info =  {'name':'Ma Yuyong', 'department':'RnD', 'studio':'SHA', 'position':'Senior Producer', 'location':'Shanghai'}
    elif 'art' in email:
        info =  {'name':'Jane Doe', 'department':'Art', 'studio':'CDU', 'position':'Art Director', 'location':'Chengdu'}
    database_meta = user_database_metadata.with_config(config).invoke({})
    if database_meta:
        info['database_meta'] = user_database_metadata.with_config(config).invoke({})
    return info

@tool
def user_database_search(query:str, config: RunnableConfig) -> list[dict]:
    """ Search for relevent documents in user's personal database. Use this for search if tavily or websearch fails or is denied.
    Returns:
        A list of dictionaries where each contains the source of the relevent document and content within the document that matches the query.
    """
    configuration = config.get("configurable", {})
    collection_id = configuration['thread_id']
    docuemnts = vector_db.retriever(collection_id).get_relevant_documents(query)
    return [{'source':d.metadata['source'],'snippet':d.page_content} for d in docuemnts]
    email = configuration.get("email", "")
    if 'angela' in email:
        return [{'filename':'qingyi.txt','snippet':'qingyi is my favourite character in ZZZ!'}]
    else:
        return [{'filename':'jane.txt','snippet':'jane is my favourite character in ZZZ!'}]
# build assistant
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    user_database_info: str

class Assistant:
    def __init__(self, prompt:ChatPromptTemplate, llm: ChatOpenAI, tools:List):
        self.prompt = prompt
        self.llm = llm
        self.tools = tools

    def runnable_with_config(self, config):
        # dont ask, only works in this order
        return self.prompt | self.llm.with_config(config).bind_tools(self.tools) 

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable_with_config(config).invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
yes_no_maybe_llm = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You simplify whaterver user say with a single 'yes' or 'no'. Reply in single word."
        ),
        ("placeholder", "{messages}"),
    ]
) | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

tools =[spell_backwards, tavily_search, user_database_search, generate_text_image]
ask_human_tool_names = {tavily_search.name, user_database_search.name, generate_text_image.name}
end_after_tool_names = {generate_text_image.name}

# llm_with_tools = llm.bind_tools(tools)

# print(llm.with_config({'configurable':{'llm':'gpt-4o'}}).bind_tools(tools).invoke('hi'))
# exit()

import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def human_said_yes(message) -> bool:
    if message.content.lower().strip() in ['ok','yes','can','1']:
        return True
    if message.content.lower().strip().startswith(('no')):
        return False
    return yes_no_maybe_llm.invoke({'messages':[message]}).content.lower() == 'yes'

# Define nodes: these do the work
ai = Assistant(
    prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant for Virtuos Games Company. "
                    " Reply using your own knowledge, only use the provided tools to search for information to assist the user's queries when asked or when you lack knowledge. "
                    " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                    " If a search comes up empty, expand your search before giving up."
                    "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                    "\nCurrent time: {time}.",
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now()),
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0).configurable_fields(
        model_name=ConfigurableField(
                id="llm/model_name",
                is_shared=True,
            ),
        temperature=ConfigurableField(
                id="llm/temperature",
                is_shared=True,
            ),
        ),
    tools=tools,
)

builder.add_node("get_user_info", lambda state:{
    'user_info':get_user_info.invoke({})
    })
builder.add_node("ai", ai)
def human_node(state: State):
    # just a placeholder for pretty graph
    return {'messages':[]}
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_node("human", human_node)
# Define edges: these determine how the control flow moves
builder.add_edge(START, "get_user_info")
builder.add_edge("get_user_info", "ai")
def route_from_ai(state: State, config: RunnableConfig) -> Literal['tools','human'] | END:
    next_node = tools_condition(state)
    if next_node == END:
        return END
    ask_human = config["configurable"].get("tools/ask_human", True)
    if ask_human:
        ai_message = state["messages"][-1]
        # This assumes single tool calls. To handle parallel tool calling, you'd want to
        # use an ANY condition
        first_tool_call_name = ai_message.tool_calls[0]["name"]
        if first_tool_call_name in ask_human_tool_names:
            return "human"
    return next_node
builder.add_conditional_edges("ai", route_from_ai)
def route_from_human(state:State) -> Literal['tools','ai']:
    next_node = tools_condition(state)
    if next_node == 'tools':
        return 'tools'
    return 'ai'
builder.add_conditional_edges("human", route_from_human)
def route_from_tools(state: State, config: RunnableConfig) -> Literal['ai'] | END:
    tool_message:ToolMessage = state["messages"][-1]
    if tool_message.name in end_after_tool_names:
        return END
    return 'ai'
builder.add_conditional_edges("tools", route_from_tools)


# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
MEMORY_URL="databases/memory.sqlite"
conn = sqlite3.connect(MEMORY_URL, check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory, 
                        interrupt_before=["human"],
                        )

graph.get_graph(xray=True).draw_mermaid_png(output_file_path='graph.png')

if __name__ == '__main__':
    thread_id  ='test5'
    vector_db.add(thread_id, r'D:\dev\ai-suite\virtuosgpt-orch\tests\resources\limei.py')
    vector_db.add(thread_id, 'https://langchain-ai.github.io/langgraph/how-tos/configuration/#base')

    config = {'configurable':{'thread_id':thread_id,'tools/ask_human':False}}
    try:
        events = graph.stream({"messages":("human","what is in my database?")}, config=config, stream_mode="updates")
        i = 0
        for event in events:
            print(f"====={i}=====")
            print(event)
            print(f"============")
            i+=1
    except:
        print(graph.get_state(config))