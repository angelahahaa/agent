import dotenv

dotenv.load_dotenv('.env')

from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from chatbot import tools
from chatbot.agents.jira import graph


config = {"configurable":{"thread_id":str(uuid4()), 'email':'pengshiya'}}
msgs = [
    HumanMessage(content="create jira ticket"),
    HumanMessage(content="create one for implementing pgadmin, fill everything else in for me."),
    HumanMessage(content="search internet for a good description for this task."),
    # HumanMessage(content="create one for implementing pgadmin"),
    # HumanMessage(content="its a task, medium priority, no labels, for description, can you use 'bamboozled' spelt backwards?"),
    # HumanMessage(content="quack"),
    # HumanMessage(content="no"),
    # HumanMessage(content="just kidddingg! i still want that ticket"),
    # HumanMessage(content="Tell me about qingyi"),
    # HumanMessage(content="用中文"),
    # HumanMessage(content="写一段关于她的小说吧。"),
]
try:
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
                msg = input("human: ")
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
except KeyboardInterrupt:
    pass
# finally:
#     print(graph.get_state(config))