

import logging
import os
from datetime import datetime
from typing import Annotated, List, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from chatbot.architecture.clarify import ClarifyState, clarify_agent_builder
from chatbot.tools import get_user_info, get_jira_tickets, create_jira_ticket, cancel_clarify_request



class JiraState(ClarifyState):
    user_info: str
tools=[get_jira_tickets]
sensitive_tools=[create_jira_ticket]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant for Virtuos Games Company. "
                    " Use the provided tools to assist the user. "
                    # "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                    "\nCurrent time: {time}."
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())

runnable = (
    prompt
    | llm.bind_tools(tools + sensitive_tools)
)
# clarify_runnable = (
#     prompt
#     | llm.bind_tools(tools + sensitive_tools + [cancel_clarify_request])
# )

memory = MemorySaver()
builder = StateGraph(JiraState)
builder.add_edge(START, 'init')
builder.add_node('init', lambda state, config: {'user_info':get_user_info.invoke(state, config)})
builder.add_edge('init', 'agent')
builder.add_node('agent', clarify_agent_builder(
    runnable=runnable,
    tools=[get_jira_tickets],
    sensitive_tools=[create_jira_ticket],
    state_schema=JiraState,
    ).compile(memory, interrupt_before='tools'))
builder.add_edge('agent', END)
graph = builder.compile(memory)
# graph = clarify_agent_builder(
#     runnable=runnable,
#     clarify_runnable=clarify_runnable,
#     tools=[get_jira_tickets],
#     sensitive_tools=[create_jira_ticket],
#     ).compile(memory)
# print(graph.nodes['agent'])
graph.get_graph(xray=True).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')