import os
import dotenv

from chatbot.database import session_db

dotenv.load_dotenv('.env')

import base64
import tempfile
# exit()
import uuid
from typing import Dict, Generator, List, Tuple

import gradio as gr
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage)

from chatbot.database import vector_db
# == agent
from chatbot.app import graph, tool_names

# == application


upload_file_prompt = "<span style=\"display:none\"> user have uploaded a document '{filename}' to session data. </span> 📁{filename}"
default_prompts = ['ok','improve answer by referenceing my database.','improve answer by including realtime internet content.']


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
                for tool_call in msg.tool_calls:
                    content = "\n".join([f"{k}={v}" for k,v in tool_call['args'].items()])
                    gr_msgs.append(gr.ChatMessage(role="assistant",content=content, metadata={'title':f"Request: {tool_call['name']}"}))
            else:
                gr_msgs.append(gr.ChatMessage(role="assistant",content=msg.content))
        elif isinstance(msg, ToolMessage):
            gr_msgs.append(gr.ChatMessage(role="assistant",content=msg.content, metadata={'title': f"Results: {msg.name}"}))
            if msg.name == 'generate_image_with_text':
                # customise how we display this
                encoded_image = msg.artifact['image']
                image_data = base64.b64decode(encoded_image)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    temp_file.write(image_data)
                    content = {"path":temp_file.name}
                gr_msgs.append(gr.ChatMessage(role="assistant",content={"path":temp_file.name}))
            else:
                pass
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



def _update_configurable(user_state, key, value):
    configurable = user_state['config']['configurable']
    if value:
        configurable[key] = value
    elif key in configurable:
        del configurable[key]
    return user_state

def _human_tool_input(snapshot, message):
    if message.lower().strip() in ['ok','y','1']:
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

def _update_sources(user_state):
    collection_name = user_state['config']['configurable']["thread_id"]
    sources = vector_db.get_sources(collection_name)
    if sources:
        title = "These are in your database:"
        return "\n".join([title] + [f" - {source}" for source in sources])
    else:
        return "Nothing here"

def _send_message(user_state, message, images, history) -> Generator[Tuple[Dict, List[gr.ChatMessage] | Dict], None, None]:
    config = user_state['config']
    ask = user_state.get('ask') or []
    messages = []
    history = history or []
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
    yield user_state, history

    snapshot = graph.get_state(config)
    if snapshot.next and 'tool' in snapshot.next[0]:
        # langchain expecting Human's confirmation (or deny)
        graph_input = _human_tool_input(snapshot, message)
    else:
        pending_messages = [HumanMessage(content=message) for message in user_state.get('pending_messages')]
        user_state['pending_messages'] = []
        graph_input = {"messages": pending_messages + messages}
    
    for _ in range(10): # max 10 interactions before we get angry
        events = graph.stream(graph_input, config, stream_mode="values")
        messages = []
        for event in events:
            messages = event.get('messages')
            if not messages:
                continue
            if not isinstance(messages, list):
                messages = [messages]
            history = _lc_to_gr_msgs(messages)
            yield user_state, history
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            # conversation is not complete, AI waiting for confirmation
            if ask and any(tool_call['name'] in set(ask) for tool_call in messages[-1].tool_calls):
                # ask human and this interactio
                history.append(gr.ChatMessage(role="assistant",
                content="⚠️ I would like to use this tool. Reply 'ok' to continue or reply with a reason why not.⚠️"))
                yield user_state, history
                break
            else:
                # continue without asking
                graph_input = None
                continue
        break
    return

def _send_message_and_clear_input(user_state, message, images, history):
    if not message and not images:
        gr.Warning("Expecting image or message.")
        yield user_state, message, images, history
        return
    try:
        generator =  _send_message(user_state, message, images, history)
        for user_state, history in generator:
            message = None
            images = None
            yield user_state, message, images, history
        return
    except Exception as e:
        gr.Warning(e)
        yield user_state, message, images, history
        return

def _add_to_user_db(user_state, path, history):
    collection_name = user_state['config']['configurable']["thread_id"]
    vector_db.add(collection_name, path)
    sources = _update_sources(user_state)

    prompt = upload_file_prompt.format(filename=path if path.startswith('http') else os.path.basename(path))
    user_state['pending_messages'].append(prompt)
    history = history or []
    history.append(gr.ChatMessage(role='user', content=prompt))
    return user_state, None, sources, history
    # gen = _send_message_no_image(user_state, upload_file_prompt.format(filename=os.path.basename(path)), history)
    # for user_state, history in gen:
    #     yield user_state, None, sources, history
    # return


def _on_session_change(user_state, session_id):
    user_state['config']['configurable']["thread_id"] = session_id
    config = user_state['config']
    msgs = graph.get_state(config).values.get('messages', [])
    history = _lc_to_gr_msgs(msgs)
    sources = _update_sources(user_state)
    return user_state, history, sources


def _send_message_no_image(user_state, message, history):
    for us, history in _send_message(user_state, message, [], history):
        yield us, history
    return

import json

session_db.initialise_database()
default_configurable = {
    'email':'angela', 
    'thread_id':None,
    'llm/model_name': 'gpt-4o',
    # 'llm/model_name': 'gpt-3.5-turbo',
    'llm/temperature': 1,
    }
default_ask = ['create_jira_ticket','search_internet']
with gr.Blocks() as demo:
    user_state = gr.State(lambda: {
        'config':{'configurable':default_configurable.copy()},
        'ask':default_ask.copy(),
        'pending_messages': [],
        })
    with gr.Row():
        with gr.Column():
            username = gr.Text(value=default_configurable['email'],label='username', visible=False)
            session_id = gr.Radio(label='conversation')
            with gr.Row():
                new_session_button = gr.Button('New Session', size='sm', min_width=60)
                clear_sessions_button = gr.Button('Clear Sessions', size='sm', min_width=60)
            with gr.Tabs():
                with gr.Tab("Image"):
                    images = gr.Gallery(type='filepath', format='png')
                with gr.Tab("My Database"):
                    sources = gr.Markdown()
                    upload_file_button = gr.UploadButton(size='sm')
                    url = gr.Text(container=False, placeholder='Add website to your database')
            with gr.Accordion("Advanced Options", open=False):
                model_name = gr.Radio(['gpt-3.5-turbo', 'gpt-4o'], value=default_configurable['llm/model_name'], label='model name')
                temperature = gr.Slider(0,1,label='temperature', value=default_configurable['llm/temperature'])
                ask_tools = gr.CheckboxGroup(choices=sorted(list(tool_names)), value=default_ask,label='Ask before using these tools')
            # debugging stuff
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type='messages', height="70vh")
            message = gr.Text(placeholder='enter message', container=False)
            with gr.Row():
                default_prompt_buttons = [
                    gr.Button(prompt, size='sm')
                for prompt in default_prompts]
                    

    # events 
    gr.on(fn=_on_user_change, triggers=[demo.load, username.submit], inputs=[user_state, username], outputs=[user_state, session_id])
    session_id.change(fn=_on_session_change, inputs=[user_state, session_id], outputs=[user_state, chatbot, sources])
    new_session_button.click(fn=_new_session, inputs=[user_state], outputs=[user_state, session_id])
    clear_sessions_button.click(fn=_clear_sessions, inputs=[user_state], outputs=[user_state, session_id])
    model_name.input(fn=lambda us, v:_update_configurable(us, 'llm/model_name', v), inputs=[user_state, model_name], outputs=[user_state])
    temperature.input(fn=lambda us, v:_update_configurable(us, 'llm/temperature', v), inputs=[user_state, temperature], outputs=[user_state])
    ask_tools.change(fn=lambda us, v:{**us, 'ask':v}, inputs=[user_state, ask_tools], outputs=[user_state])
    # ask_human.input(fn=lambda us, v:_update_configurable(us, 'tools/ask_human', v), inputs=[user_state, ask_human], outputs=[user_state])


    upload_file_button.upload(_add_to_user_db, inputs=[user_state, upload_file_button, chatbot], outputs=[user_state, upload_file_button, sources, chatbot])
    url.submit(_add_to_user_db, [user_state, url, chatbot], [user_state, url, sources, chatbot])

    # sending messages
    message.submit(
        fn=_send_message_and_clear_input, 
        inputs=[user_state, message, images, chatbot],
        outputs=[user_state, message, images, chatbot],
        )
    for btn in default_prompt_buttons:
        btn.click(
            _send_message_no_image,
            inputs=[user_state, btn, chatbot],
            outputs=[user_state, chatbot],
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
if __name__ == '__main__':
    demo.launch()