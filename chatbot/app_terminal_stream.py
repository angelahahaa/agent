import json
from typing import List, Set

import dotenv

dotenv.load_dotenv('.env')

import asyncio
from uuid import uuid4

import aioconsole
from langchain_core.messages import (AIMessageChunk, BaseMessage, HumanMessage,
                                     ToolMessage, ToolCallChunk)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from chatbot import tools
from chatbot.agents.all_multiagent import graph
from chatbot.architecture.multiagent import Agent, multi_agent_builder
from chatbot.mocks import MockChat, mock_tool

print(graph.graph_schema().schema())
exit()
def print_event(event):
    print(f"==== {event['event']:<20} : {event['name']:<20} ====")


def print_tool_calls(tool_calls):
    for tc in tool_calls:
        args = ", ".join([f"{k}={v}" for k,v in tc['args'].items()])
        print(f"{'':10} <tool_call> {tc['name']}({args})")

def print_messages_without_duplicates(messages:List[BaseMessage], id_tracker:Set|None=None):
    id_tracker = set() if id_tracker is None else id_tracker
    for message in messages:
        if message.id not in id_tracker:
            id_tracker.add(message.id)
            print(f"{message.type:>10}: {message.content.replace('\n',f"{'\n':10}")}")
            if message.type == 'ai' and message.tool_calls:
                print_tool_calls(message.tool_calls)

async def main():
    config = {"configurable":{"thread_id":str(uuid4()), 'email':'pengshiya'}}
    msgs = [
        HumanMessage(content="hi."),
    ]
    try:
        for _ in range(10): 
            snapshot = graph.get_state(config)
            if 'tools' in snapshot.next:
                approve = ''
                while approve.lower() not in ['y','n']:
                    approve = await aioconsole.ainput("approve (y/n):")
                    approve = approve.strip()
                if approve.lower() == 'y':
                    inputs = None
                else:
                    reason = await aioconsole.ainput("reason (optional):")
                    inputs = {"messages":[
                        ToolMessage(content=f"API call denied by user. Reasoning: '{reason}'. Continue assisting, accounting for the user's input.", tool_call_id=tc['id'])
                        for tc in snapshot.values['messages'][-1].tool_calls
                        ]}
            else:
                if msgs:
                    msg = msgs.pop(0)
                    print(f"human: {msg}")
                else:
                    msg = HumanMessage(content=input("human: "))
                inputs = {"messages":[msg]}
            events = graph.astream_events(
                inputs, config, 
                version='v2',
                )
            async for event in events:
                if event['event'] == "on_chat_model_start":
                    print_event(event)
                elif event['event'] == "on_chat_model_stream":
                    chunk:AIMessageChunk = event['data']["chunk"]
                    print(chunk.content, end='')
                if event['event'] == "on_chat_model_end":
                    print('')
                    print_event(event)
                    print_tool_calls(event['data']['output'].tool_calls)
                elif event['event'] == "on_tool_end":
                    print_event(event)
                    print(event['data']['output'])
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print("======================================")
        # print(graph.get_state(config))
        print_messages_without_duplicates(graph.get_state(config).values['messages'])
        print("======================================")
asyncio.run(main())