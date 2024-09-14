

import os
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnablePassthrough
from chatbot.architecture.clarify import clarify_agent_builder
from chatbot.tools import (create_jira_ticket,
                           get_jira_tickets, search_internet)
from chatbot.mocks import MockChat
class Agent:
    def __init__(self, llm:BaseChatModel):
        self.llm = llm
    def bind_tools(self, tools):
        return (
            RunnablePassthrough.assign(**{"time":lambda x:datetime.now()})
            | ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a helpful assistant for Virtuos Games Company. "
                    " Use the provided tools to assist the user. "
                    "\nCurrent time: {time}."
                ),
                ("placeholder", "{messages}"),
                ])
            | self.llm.bind_tools(tools))

tools=[get_jira_tickets]
sensitive_tools=[create_jira_ticket, search_internet]
all_tools = tools + sensitive_tools
memory = MemorySaver()
graph = clarify_agent_builder(
    runnable=Agent(ChatOpenAI(model="gpt-4o-mini", temperature=0)),
    # runnable=Agent(MockChat(model="gpt-4o-mini", temperature=0)),
    tools=tools,
    sensitive_tools=sensitive_tools,
    ).compile(memory)
graph.get_graph(
    # xray=True,
    ).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')
