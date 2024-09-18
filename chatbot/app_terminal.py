from typing import List, Set
import dotenv

dotenv.load_dotenv('.env')

from uuid import uuid4

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from chatbot import tools
from chatbot.agents.all_in_one import graph

def print_messages_without_duplicates(messages:List[BaseMessage], id_tracker:Set|None=None):
    id_tracker = set() if id_tracker is None else id_tracker
    for message in messages:
        if message.id not in id_tracker:
            id_tracker.add(message.id)
            print(f"{message.type:>10}: {message.content.replace('\n',f"{'\n':10}")}")
            if message.type == 'ai' and message.tool_calls:
                for tc in message.tool_calls:
                    args = ", ".join([f"{k}={v}" for k,v in tc['args'].items()])
                    print(f"{'':10} <tool_call> {tc['name']}({args})")
    

config = {"configurable":{"thread_id":str(uuid4()), 'email':'pengshiya'}}
msgs = [
    # HumanMessage(content="What tasks do I have?"),
    # HumanMessage(content="Great, thanks! I need some inspiration for my designs. Could you help me find some references about cyberpunk?"),
    # HumanMessage(content="I need to install substance painter."),
    # HumanMessage(content="quack"),
    # HumanMessage(content="create jira ticket"),
    # HumanMessage(content="create one for implementing pgadmin, fill everything else in for me."),
    # HumanMessage(content="search internet for a good description for this task."),
    # HumanMessage(content="create one for implementing pgadmin"),
    # HumanMessage(content="its a task, medium priority, no labels, for description, can you use 'bamboozled' spelt backwards?"),
    # HumanMessage(content="quack"),
    # HumanMessage(content="可以可以，你整吧。"),
    # HumanMessage(content="no"),
    # HumanMessage(content="just kidddingg! i still want that ticket"),
    # HumanMessage(content="Tell me about qingyi"),
    # HumanMessage(content="用中文"),
    # HumanMessage(content="写一段关于她的小说吧。"),
]
try:
    printed_message_id = set()
    for _ in range(10): 
        if graph.get_state(config).next:
            approve = input("approve:")
            inputs = None
        else:
            # msg = f"" 
            # print("human: ")
            if msgs:
                msg = msgs.pop(0)
                print(f"human: {msg}")
            else:
                msg = HumanMessage(content=input("human: "))
            inputs = {"messages":[msg]}
        
        events = graph.stream(inputs, config, stream_mode='updates')
        for updates in events:
            for node, updates in updates.items():
                print(f"============ {node} ============")
                for k, v in updates.items():
                    if k in ['messages']:
                        print(f"{k}:")
                        # print(v)
                        print_messages_without_duplicates(v, printed_message_id)
                    else:
                        print(f"{k}: {v}")
                    print()
except KeyboardInterrupt:
    pass
finally:
    print()
    print("======================================")
    # print(graph.get_state(config))
    print_messages_without_duplicates(graph.get_state(config).values['messages'])
    print("======================================")
