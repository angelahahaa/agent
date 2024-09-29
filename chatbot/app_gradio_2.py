from collections import defaultdict
import json
import random
import time
from typing import Dict, Iterator, List, Literal, NotRequired
from typing_extensions import TypedDict
from uuid import uuid4
import gradio as gr
from dataclasses import dataclass, field

from typing import Dict, List
import gradio as gr



# fake gpt
responses = [
"""event: message_start

data: {"content": "Hello", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": "!", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": " How", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": " can", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": " I", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": " assist", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": " you", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": " today", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"content": "?", "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "type": "ai"}

data: {"type": "ai", "name": null, "id": "run-a6a84c4a-e8f6-4187-b9bd-6dc5f0a60ae0", "tool_calls": []}

event: message_end

""",
"""event: message_start

data: {"type": "ai", "name": null, "id": "run-1bcef17a-dc92-4af1-ba68-8631f360bcdf", "tool_calls": [{"name": "accept_this_one", "args": {}, "id": "call_SabmtRGYklDvSDYGjAcIE9J5", "type": "tool_call"}, {"name": "deny_this_one", "args": {}, "id": "call_SabmtRGYklDvSDYGjAcIE9J4", "type": "tool_call"}]}

event: message_end

""",
"""event: message_start

data: {"content": "{}", "type": "tool", "name": "accept_this_one", "id": null, "tool_call_id": "call_SabmtRGYklDvSDYGjAcIE9J5", "artifact": null, "status": "success"}

event: message_end

event: message_start

data: {"content": "It", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " looks", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " like", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " there", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " isn't", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " any", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " specific", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " user", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " information", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " available", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": ".", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " If", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " you", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " have", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " particular", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " details", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " or", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " questions", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " about", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " yourself", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " that", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " you'd", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " like", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " to", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " share", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " or", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " ask", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": ",", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " feel", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": " free", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"content": "!", "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "type": "ai"}

data: {"type": "ai", "name": null, "id": "run-b258017b-58a4-464f-85dc-413d7dcc85c1", "tool_calls": []}

event: message_end

"""
]
class EventStream(TypedDict):
    event:NotRequired[str]
    data:NotRequired[Dict]
def chat_stream(input:Dict | None) -> Iterator[EventStream]:
    if input is None:
        lines = responses[2]
    else:
        lines = responses[random.randint(0,1)]
    event = EventStream()
    for line in lines.splitlines(keepends=True):
        line = line.rstrip('\n')
        if not line:
            time.sleep(0.1)
            yield event
            event = EventStream()
        elif line.startswith('event: '):
            event['event'] = line[7:]
        elif line.startswith('data: '):
            event['data'] = json.loads(line[6:])

def chat_deny(reason:str, id:str):
    return {
        'name': 'deny_this_one',
        'content': f"Tool call denied by user. {reason}",
        'tool_call_id': id,
        'artifact': {'reason': reason},
        'type':'tool',
    }

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
    tool_calls: List[Dict] = field(default_factory=list)

with gr.Blocks() as demo:
    with gr.Accordion('hidden', open=False, render=False) as hidden_accordion:
        user_state = gr.State(UserState)
        refresh_tool_call_trigger = gr.Number(1)
    messages_chatbot = gr.Chatbot(type='messages')
    with gr.Row():
        text_text = gr.Text(container=False, scale=4)
        send_button = gr.Button('send', size='sm', min_width=60)


    def send(user:UserState, input:Dict|None):
        for event_stream in chat_stream(input):
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
        user.tool_calls = user.messages[-1].get('tool_calls',[])
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
            text_text:None, 
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
        visible = not bool(user.tool_calls)
        return gr.update(visible=visible), gr.update(visible=visible)

    @gr.render(
        triggers=[refresh_tool_call_trigger.change],
        inputs=[user_state],
    )
    def render_tool_calls(user:UserState):
        if not user.tool_calls:
            return
        tool_call_components = []
        with gr.Row() as tool_call_group:
            for tool_call in user.tool_calls:
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
            user = inputs[user_state]
            for tool_call_json, decision_radio, reason_text in tool_call_components:
                decision = inputs[decision_radio]
                tc = inputs[tool_call_json]
                reason = inputs[reason_text]
                if decision =='deny':
                    user.messages.append(chat_deny(reason, tc['id']))
                    yield {messages_chatbot: to_chat_messages(user.messages)}
            user.tool_calls = []
            yield {
                tool_call_group: gr.update(visible=False), 
                submit_button:gr.update(visible=False),
                text_text:gr.update(visible=True),
                send_button:gr.update(visible=True),
                }
            for d in send(user, None):
                yield d

    hidden_accordion.render()
    with gr.Accordion('debug', open=False):
        gr.Button('user_state').click(lambda x:print(f"====\n{json.dumps(x.messages, indent=2)}\n{json.dumps(x.tool_calls)}\n===="), inputs=[user_state])

demo.launch()