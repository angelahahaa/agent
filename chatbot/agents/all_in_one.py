

import os
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnablePassthrough
from chatbot.architecture.clarify import clarify_agent_builder
from chatbot import tools
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
                    " Prioritise searching user uploaded content or internal database over the internet. "
                    "\nCurrent time: {time}."
                ),
                ("placeholder", "{messages}"),
                ])
            | self.llm.bind_tools(tools))

normal_tools=[
    tools.search_IT_procedure,
    tools.search_user_projects,
    # tools.search_session_data,
    tools.search_internet,
    tools.get_jira_tickets,
    tools.generate_images,
    ]
sensitive_tools=[
    tools.create_jira_ticket,
    tools.create_IT_fresh_service_ticket,
    ]
all_tools = normal_tools + sensitive_tools
memory = MemorySaver()
graph = clarify_agent_builder(
    runnable=Agent(ChatOpenAI(model="gpt-4o-mini", temperature=0)),
    # runnable=Agent(MockChat(model="gpt-4o-mini", temperature=0)),
    tools=normal_tools,
    sensitive_tools=sensitive_tools,
    ).compile(memory)
graph.get_graph(
    # xray=True,
    ).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')
