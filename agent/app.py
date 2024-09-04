from datetime import datetime
import dotenv
dotenv.load_dotenv('.env')
from typing import TypedDict, List, Literal, Dict, Any

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools.tavily_search import TavilySearchResults
# helpers
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage, HumanMessage, RemoveMessage

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
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
tavily_search = TavilySearchResults(max_results=1)

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
    if ('angela' in email) or ('shiya' in email):
        return {'name':'Peng Shiya', 'department':'RnD', 'studio':'SHA', 'position':'Engineer', 'location':'Shanghai'}
    elif 'pengseng' in email:
        return {'name':'Ang Peng Seng', 'department':'RnD', 'studio':'SGP', 'position':'Lead Engineer', 'location':'Singapore'}
    elif 'yuyong' in email:
        return {'name':'Ma Yuyong', 'department':'RnD', 'studio':'SHA', 'position':'Senior Producer', 'location':'Shanghai'}
    elif 'art' in email:
        return {'name':'Jane Doe', 'department':'Art', 'studio':'CDU', 'position':'Art Director', 'location':'Chengdu'}
    return dict()

@tool
def user_database_search(query:str, config: RunnableConfig) -> list[dict]:
    """ Search for relevent documents in user's personal database. Use this for search if tavily or websearch fails or is denied.
    Returns:
        A list of dictionaries where each contains the filename of the relevent document and content within the document that matches the query.
    """
    configuration = config.get("configurable", {})
    email = configuration.get("email", "")
    if 'angela' in email:
        return [{'filename':'qingyi.txt','snippet':'qingyi is my favourite character in ZZZ!'}]
    else:
        return [{'filename':'jane.txt','snippet':'jane is my favourite character in ZZZ!'}]
# build assistant
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
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
    
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
yes_no_maybe_llm = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You simplify whaterver user say with a single 'yes' or 'no'. Reply in single word."
        ),
        ("placeholder", "{messages}"),
    ]
) | llm
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for Virtuos. "
            " Use the provided tools to search for information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
tools = [spell_backwards, tavily_search, user_database_search]
ask_human_tool_names = {tavily_search.name, user_database_search.name}

assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)


def human_said_yes(message) -> bool:
    if message.content.lower().strip() in ['ok','yes','can','1']:
        return True
    if message.content.lower().strip().startswith(('no')):
        return False
    return yes_no_maybe_llm.invoke({'messages':[message]}).content.lower() == 'yes'

# Define nodes: these do the work
builder.add_node("get_user_info", lambda state:{'user_info':get_user_info.invoke({})})
builder.add_node("ai", Assistant(assistant_runnable))
def human_node(state: State):
    # just a placeholder for pretty graph
    return {'messages':[]}
builder.add_node("tools", create_tool_node_with_fallback(tools))
builder.add_node("human", human_node)
# Define edges: these determine how the control flow moves
builder.add_edge(START, "get_user_info")
builder.add_edge("get_user_info", "ai")
def route_from_ai(state: State) -> Literal['tools','human'] | END:
    next_node = tools_condition(state)
    if next_node == END:
        return END
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
builder.add_edge("tools", "ai")
builder.add_conditional_edges("human", route_from_human)


# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory, 
                        interrupt_before=["human"],
                        )
from PIL import Image
graph.get_graph(xray=True).draw_mermaid_png(output_file_path='graph.png')
# exit()
import uuid


from agent import database
import gradio as gr
from typing import TypedDict, List, Literal
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage, HumanMessage, RemoveMessage

def _lc_to_gr_msgs(lc_msgs:List[BaseMessage]) -> List[gr.ChatMessage]:
    gr_msgs = []
    for msg in lc_msgs:
        if isinstance(msg, HumanMessage):
            gr_msgs.append(gr.ChatMessage(role="user",content=msg.content))
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_name = msg.tool_calls[0]['name']
                gr_msgs.append(gr.ChatMessage(role="assistant",content=json.dumps(msg.tool_calls), metadata={'title':f"Let me call {tool_name}"}))
            else:
                gr_msgs.append(gr.ChatMessage(role="assistant",content=msg.content))
        elif isinstance(msg, ToolMessage):
            gr_msgs.append(gr.ChatMessage(role="assistant",content=msg.content, metadata={'title': f"{msg.name} results"}))
        elif isinstance(msg, SystemMessage):
            ...
    return gr_msgs

def _new_session(user_state):
    new_session_id = str(uuid.uuid4())
    username = user_state['config']['configurable']["email"]
    database.add_session(session_id=new_session_id, username=username)
    sessions = database.get_active_session_ids(username)
    return user_state, gr.Radio(choices = sessions, value=sessions[0])

def _on_user_change(user_state, username):
    sessions = database.get_active_session_ids(username)
    value = sessions[0] if sessions else None
    user_state['config']['configurable']["email"] = username
    user_state['config']['configurable']["thread_id"] = value
    return user_state, gr.Radio(choices = sessions, value=value)

def _on_session_change(user_state, session_id):
    user_state['config']['configurable']["thread_id"] = session_id
    config = user_state['config']
    msgs = graph.get_state(config).values.get('messages', [])
    history = _lc_to_gr_msgs(msgs)
    return user_state, history

def _send_message(user_state, message, history):
    config = user_state['config']

    human_message = HumanMessage(message)
    history.extend(_lc_to_gr_msgs([human_message]))
    yield user_state, "", history
    # get previous state
    snapshot = graph.get_state(config)
    if snapshot.next:
        if message.lower().strip() == 'ok':
            graph_input = None
        else:
            tool_call_id = snapshot.values['messages'][-1].tool_calls[0]["id"]
            graph_input = {
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=f"API call denied by user. Reasoning: '{message}'. Continue assisting, accounting for the user's input.",
                            name='user_denied'
                        )
                    ]
                }
    else:
        graph_input = {"messages": human_message}
    
    history.extend(_lc_to_gr_msgs([human_message]))
    yield user_state, "", history
    events = graph.stream(graph_input, config, stream_mode="values")
    for event in events:
        messages = event.get('messages')
        if not messages:
            continue
        if not isinstance(messages, list):
            messages = [messages]
        yield user_state, "", _lc_to_gr_msgs(messages)
    return
import json

database.initialise_database()
with gr.Blocks() as demo:
    user_state = gr.State(lambda: {'config':{'configurable':{'email':None, 'thread_id':None}}})
    with gr.Row():
        with gr.Column():
            username = gr.Text('angela',label='username')
            session_id = gr.Radio(label='session')
            new_session_button = gr.Button('New Session')
            # model = gr.Dropdown(['chatgpt-3.5', 'chatgpt-4o'],value='chatgpt-3.5',label='model')
            # debugging stuff
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type='messages', height="80vh")
            message = gr.Text(placeholder='enter message', container=False)

    # events 
    gr.on(fn=_on_user_change, triggers=[demo.load, username.submit], inputs=[user_state, username], outputs=[user_state, session_id])
    session_id.change(fn=_on_session_change, inputs=[user_state, session_id], outputs=[user_state, chatbot])
    message.submit(fn=_send_message, inputs=[user_state, message, chatbot],outputs=[user_state, message, chatbot])
    new_session_button.click(fn=_new_session, inputs=[user_state], outputs=[user_state, session_id])

    # debug stuff
    with gr.Accordion("Debug",open=True):
        with gr.Row():
            us_btn = gr.Button('Print user_state')
            graph_state_btn = gr.Button('Print graph state')
        console = gr.Textbox(container=False)
    us_btn.click(fn=lambda x:json.dumps(x, indent=2, default=str), inputs=[user_state], outputs=[console])
    @graph_state_btn.click(inputs=[user_state], outputs=[console])
    def _fn(user_state):
        return json.dumps(graph.get_state(user_state['config']), indent=2, default=str)
demo.launch()