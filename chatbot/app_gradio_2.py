from datetime import datetime
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, NotRequired
from uuid import uuid4

import gradio as gr
import httpx
from typing_extensions import TypedDict

from chatbot.client import ChatbotAPI


def update_message(message:Dict, chunk:Dict):
    for k,v in chunk.items():
        if k in ['id','type'] and k in message:
            assert message.get(k) == v
            continue
        elif k in ['content','tool_calls'] and k in message:
            message[k] += v
            continue
        message[k] = v

# ===    

def to_chat_messages(messages:List[Dict]) -> List[gr.ChatMessage]:
    outputs:List[gr.ChatMessage] = []
    for msg in messages:
        if not (msg_type:=msg.get('type')):
            continue
        if msg_type == 'human':
            outputs.append(gr.ChatMessage(role='user', content=msg['content']))
        elif msg_type == 'ai':
            if content:=msg.get('content'):
                outputs.append(gr.ChatMessage(role='assistant', content=content))
            if tool_calls:=msg.get('tool_calls'):
                pass
        elif msg_type == 'tool':
            outputs.append(gr.ChatMessage(role='assistant', content=msg['content'], metadata={'title':msg['name']}))
                
    return outputs

@dataclass
class UserState:
    messages: List[Dict] = field(default_factory=list)
    session_id: str | None = None

client = ChatbotAPI("http://localhost:1234/agent", token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbmdlbGFAYWJjLmNvbSIsImV4cCI6MTcyNzM0MjcxNH0.JHO8i-dQLD_ooL3-7AFV3b63m6l7OttK_jzky5dV9HI")
agent_name_choices = client.get_available_agents()

with gr.Blocks() as demo:
    # Components
    user_state = gr.State(UserState, render=False)
    refresh_tool_call_trigger = gr.Number(1, label='refresh_tool_call_trigger', render=False)
    messages_chatbot = gr.Chatbot(type='messages', render=False, height="70vh", show_label=False)
    text_text = gr.Text(container=False, render=False)
    send_button = gr.Button('send', size='sm', min_width=60, render=False)

    sessions_radio = gr.Radio([], label='sessions', container=False, render=False)
    new_session_button = gr.Button('New Session', size='sm', min_width=60, render=False)
    delete_session_button = gr.Button('Delete Session', size='sm', min_width=60, render=False)
    agent_name_dropdown = gr.Dropdown(agent_name_choices, value=agent_name_choices[0], container=False, render=False)

    # Events
    @demo.load(outputs=[sessions_radio])
    def update_sessions():
        sessions = client.get_sessions()
        sessions.sort(key=lambda x: datetime.fromisoformat(x['last_modified']), reverse=True)
        sessions = [(f"{s['agent_name']} ({s['session_id']})", s['session_id']) for s in sessions]
        return {sessions_radio:gr.Radio(sessions, value=sessions[0][1] if sessions else None)}

    @new_session_button.click(inputs=[agent_name_dropdown], outputs=[sessions_radio])
    def new_session(agent_name:str):
        client.create_session(agent_name)
        return update_sessions()

    @delete_session_button.click(inputs=[sessions_radio], outputs=[sessions_radio])
    def delete_session(session_id:str):
        client.archive_session(session_id)
        return update_sessions()

    @sessions_radio.change(inputs=[user_state, sessions_radio], outputs=[user_state, messages_chatbot, refresh_tool_call_trigger])
    def on_session_change(user:UserState, session_id:str|None):
        user.session_id = session_id
        if session_id is None:
            return {messages_chatbot:[]}
        user.messages = client.get_chat_history(session_id)['messages']
        yield {user_state: user}
        yield {messages_chatbot:to_chat_messages(user.messages), refresh_tool_call_trigger:random.random()}

    def send(user:UserState, input:Dict | None):
        if not user.session_id:
            raise gr.Error('No active session')
        for event_stream in client.chat_stream(user.session_id, input, {}):
            event = event_stream.get('event')
            data = event_stream.get('data', {})
            if event == 'error':
                print(data['error'])
                raise
            elif event == 'message_start':
                user.messages.append({})
            elif event == 'message_end':
                ...
            elif event is None:
                update_message(user.messages[-1], data)
                yield  {
                    user_state: user, 
                    messages_chatbot:to_chat_messages(user.messages),
                    }
        yield {user_state:user}
        yield {refresh_tool_call_trigger:random.random()}

    @gr.on(
        triggers=[text_text.submit, send_button.click],
        inputs=[user_state, text_text],
        outputs=[user_state, text_text, messages_chatbot, refresh_tool_call_trigger],
    )
    def run_send_with_user_message(user:UserState, text:str):
        message = {
                'type':'human',
                'content':text,
            }
        user.messages.append(message)
        yield  {
            text_text: gr.Text(None), 
            user_state: user, 
            messages_chatbot:to_chat_messages(user.messages),
            }
        input = {
            'messages': [message]
        }
        for d in send(user, input):
            yield d

    @refresh_tool_call_trigger.change(
        inputs=[user_state],
        outputs=[text_text, send_button],
    )
    def toggle_user_text(user:UserState):
        if not user.messages:
            visible = True
        message = user.messages[-1]
        visible = not bool(message.get('tool_calls'))
        return gr.update(visible=visible), gr.update(visible=visible)



    # UI
    with gr.Row():
        agent_name_dropdown.render()
        new_session_button.render()
        delete_session_button.render()
    sessions_radio.render()
    messages_chatbot.render()
    with gr.Row():
        text_text.scale=4
        text_text.render()
        send_button.render()
    @gr.render(
        triggers=[refresh_tool_call_trigger.change],
        inputs=[user_state],
    )
    def render_tool_calls(user:UserState):
        if not (tool_calls:=user.messages[-1].get('tool_calls')):
            return
        tool_call_components = []
        with gr.Row() as tool_call_group:
            for tool_call in tool_calls:
                with gr.Column(variant='panel'):
                    tool_call_json = gr.JSON(tool_call)
                    decision_radio = gr.Radio(['approve','deny'], value='approve', container=False)
                    reason_text = gr.Text(container=False, placeholder='reason (optional)', visible=False)
                    decision_radio.change(lambda x: gr.update(visible=x=='deny'), inputs=[decision_radio], outputs=[reason_text])
                    tool_call_components.append([tool_call_json, decision_radio, reason_text])
        submit_button = gr.Button('submit', size='sm')
        @submit_button.click(
            inputs={c for components in tool_call_components for c in components} | {user_state},
            outputs=[user_state, refresh_tool_call_trigger, messages_chatbot, tool_call_group, submit_button, text_text, send_button]
        )
        def submit_tool_call_decisions(inputs):
            user:UserState = inputs[user_state]
            messages = []
            for tool_call_json, decision_radio, reason_text in tool_call_components:
                decision = inputs[decision_radio]
                tc = inputs[tool_call_json]
                reason = inputs[reason_text]
                if decision =='deny':
                    messages.append({"type": "tool", "tool_call_id": tc['id'],"reason":reason})
            if messages:
                if not user.session_id:
                    raise gr.Error('No active session')
                user.messages.extend(client.chat_tools_deny(user.session_id, {"messages":messages})['messages'])
                yield {messages_chatbot: to_chat_messages(user.messages)}
            yield {
                tool_call_group: gr.update(visible=False), 
                submit_button:gr.update(visible=False),
                text_text:gr.update(visible=True),
                send_button:gr.update(visible=True),
                }
            for d in send(user, None):
                yield d
    with gr.Accordion('debug', open=False):
        user_state.render()
        refresh_tool_call_trigger.render()
        gr.Button('user_state').click(lambda x:print(f"====\n{json.dumps(x.messages, indent=2)}\n===="), inputs=[user_state])
        gr.Button('send_empty').click(lambda us: client.chat_invoke(us.session_id, None, None), inputs=[user_state])


demo.launch()