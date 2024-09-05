import dotenv

dotenv.load_dotenv('.env')

import base64
import tempfile
# exit()
import uuid
from typing import List

import gradio as gr
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)

from agent import session_db
# == agent
from agent.chatbot import graph

# == file upload




def _lc_to_gr_msgs(lc_msgs:List[BaseMessage]) -> List[gr.ChatMessage]:
    gr_msgs = []
    for msg in lc_msgs:
        if isinstance(msg, HumanMessage):
            contents = []
            if isinstance(msg.content, str):
                contents.append(msg.content)
            elif isinstance(msg.content, list):
                for c in msg.content:
                    if isinstance(c, str):
                        contents.append(c)
                    elif isinstance(c, dict):
                        if c.get("type") == 'text':
                            contents.append(c.get("text"))
                        elif c.get("type") == 'image_url':
                            # assume it is always image bytes
                            encoded_image = c["image_url"]['url'].split(",")[1]
                            image_data = base64.b64decode(encoded_image)

                            # Write the image data to a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                                temp_file.write(image_data)
                                contents.append({"path":temp_file.name})
                            
            gr_msgs.extend([gr.ChatMessage(role="user",content=content) for content in contents])
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_name = msg.tool_calls[0]['name']
                gr_msgs.append(gr.ChatMessage(role="assistant",content=json.dumps(msg.tool_calls), metadata={'title':f"Request: {tool_name}"}))
            else:
                gr_msgs.append(gr.ChatMessage(role="assistant",content=msg.content))
        elif isinstance(msg, ToolMessage):
            gr_msgs.append(gr.ChatMessage(role="assistant",content=msg.content, metadata={'title': f"Results: {msg.name}"}))
        elif isinstance(msg, SystemMessage):
            ...
    return gr_msgs

def _new_session(user_state):
    new_session_id = str(uuid.uuid4())
    username = user_state['config']['configurable']["email"]
    session_db.add(session_id=new_session_id, username=username)
    sessions = session_db.get_sessions(username)
    return user_state, gr.Radio(choices = sessions, value=sessions[0])

def _clear_sessions(user_state):
    username = user_state['config']['configurable']["email"]
    sessions = session_db.get_sessions(username)
    for session in sessions:
        session_db.archive(session)
    return _new_session(user_state)

def _on_user_change(user_state, username):
    sessions = session_db.get_sessions(username)
    user_state['config']['configurable']["email"] = username
    if not sessions:
        return _new_session(user_state)
    return user_state, gr.Radio(choices = sessions, value=sessions[0])

def _on_session_change(user_state, session_id):
    user_state['config']['configurable']["thread_id"] = session_id
    config = user_state['config']
    msgs = graph.get_state(config).values.get('messages', [])
    history = _lc_to_gr_msgs(msgs)
    return user_state, history


def _update_configurable(user_state, key, value):
    configurable = user_state['config']['configurable']
    if value:
        configurable[key] = value
    elif key in configurable:
        del configurable[key]
    return user_state

def _human_tool_input(snapshot, message):
    if message.lower().strip() == 'ok':
        return None
    else:
        tool_call_id = snapshot.values['messages'][-1].tool_calls[0]["id"]
        return {
                "messages": [
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=f"API call denied by user. Reasoning: '{message}'. Continue assisting, accounting for the user's input.",
                        name='user_denied'
                    )
                ]
            }

def _send_message(user_state, message, images, history):
    config = user_state['config']
    messages = []
    if not images and not message:
        gr.Warning('Nothing to send')
        return user_state, gr.update(), gr.update(), gr.update()
    if images:
        for image in images:
            with open(image[0], 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            messages.append(HumanMessage(content=[{
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/png;base64,{encoded_image}",
                    "detail":"low", # TODO: make this an option
                    },
            }]))
    if message:
        messages.append(HumanMessage(message))
    history.extend(_lc_to_gr_msgs(messages))
    yield user_state, "", None, history

    snapshot = graph.get_state(config)
    if "human" in snapshot.next:
        # langchain expecting Human's confirmation (or deny)
        graph_input = _human_tool_input(snapshot, message)
    else:
        graph_input = {"messages": messages}
    
    events = graph.stream(graph_input, config, stream_mode="values")
    for event in events:
        messages = event.get('messages')
        if not messages:
            continue
        if not isinstance(messages, list):
            messages = [messages]
        yield user_state, gr.update(), gr.update(), _lc_to_gr_msgs(messages)
    return
import json

session_db.initialise_database()
default_configurable = {
    'email':'angela', 
    'thread_id':None,
    # 'llm/model_name': 'gpt-4o',
    'llm/model_name': 'gpt-3.5-turbo',
    'llm/temperature': 1,
    'tools/ask_human': True,
    }
with gr.Blocks() as demo:
    user_state = gr.State(lambda: {'config':{'configurable':default_configurable.copy()}})
    with gr.Row():
        with gr.Column():
            username = gr.Text(value=default_configurable['email'],label='username')
            session_id = gr.Radio(label='session')
            with gr.Row():
                new_session_button = gr.Button('New Session', size='sm', min_width=60)
                clear_sessions_button = gr.Button('Clear Sessions', size='sm', min_width=60)
            with gr.Tabs():
                with gr.Tab("Image"):
                    images = gr.Gallery(type='filepath', format='png')
                # with gr.Tab("My Database"):
                #     files = gr.Files()
            with gr.Accordion("Advanced Options", open=False):
                model_name = gr.Radio(['gpt-3.5-turbo', 'gpt-4o'], value=default_configurable['llm/model_name'], label='model name')
                temperature = gr.Slider(0,1,label='temperature', value=default_configurable['llm/temperature'])
                ask_human = gr.Checkbox(value=default_configurable['tools/ask_human'], label='ask_human')
            # debugging stuff
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type='messages', height="70vh")
            message = gr.Text(placeholder='enter message', container=False)

    # events 
    gr.on(fn=_on_user_change, triggers=[demo.load, username.submit], inputs=[user_state, username], outputs=[user_state, session_id])
    session_id.change(fn=_on_session_change, inputs=[user_state, session_id], outputs=[user_state, chatbot])
    new_session_button.click(fn=_new_session, inputs=[user_state], outputs=[user_state, session_id])
    clear_sessions_button.click(fn=_clear_sessions, inputs=[user_state], outputs=[user_state, session_id])
    model_name.input(fn=lambda us, v:_update_configurable(us, 'llm/model_name', v), inputs=[user_state, model_name], outputs=[user_state])
    temperature.input(fn=lambda us, v:_update_configurable(us, 'llm/temperature', v), inputs=[user_state, temperature], outputs=[user_state])
    ask_human.input(fn=lambda us, v:_update_configurable(us, 'llm/ask_human', v), inputs=[user_state, ask_human], outputs=[user_state])

    message.submit(
        fn=_send_message, 
        inputs=[user_state, message, images, chatbot],
        outputs=[user_state, message, images, chatbot],
        )

    # debug stuff
    with gr.Accordion("Debug",open=False):
        with gr.Row():
            us_btn = gr.Button('Print user_state', size='sm')
            graph_state_btn = gr.Button('Print graph state', size='sm')
        console = gr.Textbox(container=False)
    us_btn.click(fn=lambda x:json.dumps(x, indent=2, default=str), inputs=[user_state], outputs=[console])
    @graph_state_btn.click(inputs=[user_state], outputs=[console])
    def _fn(user_state):
        return json.dumps(graph.get_state(user_state['config']), indent=2, default=str)
demo.launch()