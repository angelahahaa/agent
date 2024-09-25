from typing import List, Set

import dotenv

dotenv.load_dotenv('.env')


import logging
import os
import random
import time
from datetime import datetime
from typing import (Annotated, Dict, Generic, List, Literal, NotRequired, Set,
                    TypedDict, TypeVar)
from uuid import uuid4

from langchain.chat_models.base import BaseChatModel
from langchain.tools.base import StructuredTool
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from typing_extensions import TypedDict

from chatbot import tools
from chatbot.architecture.clarify import clarify_agent_builder
from chatbot.architecture.multiagent import Agent, multi_agent_builder, MultiAgentState
from chatbot.mocks import MockChat, mock_tool


@tool
def complete_or_escalate(reason:str, state: Annotated[MultiAgentState, InjectedState]):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""
    current_agent = state['current_agent']
    if len(current_agent) > 2:
        name = current_agent[-2]
    else:
        name = 'primary'
    return f"Resuming dialog with the {name} assistant. Please reflect on the past conversation and assist the user as needed."

@tool
def enter_jira_agent(task:str):
    """Transfers work to 'jira' assistant to handle jira related tasks.
    This agent is capable of:
    - Creating new issues in JIRA for project tracking.
    - Updating existing JIRA issues with new information or status changes.
    - Tracking and reporting on the progress of JIRA issues.
    - Checking and providing updates on project statuses.
    """
    return f"The assistant is now the 'jira' assistent with a set of jira tools."\
            f" The user's intent is unsatisfied. Use the provided tools to complete the task."

shared_tools = [
        tools.get_user_info,
        tools.search_session_data,
        tools.get_user_projects,
]

jira_agent = Agent(
    name='jira',
    prompt=(
        RunnablePassthrough.assign(**{"time":lambda x:datetime.now()})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a specialised {agent_name} assistant for Virtuos Games Company. "
                " Use the provided tools to assist the user. "
                " Prioritise searching user uploaded content or internal database over the internet. "
                " Please confirm all details before creating a JIRA ticket. You can suggest values based on the conversation."
                " Always ask for confirmation before proceeding with JIRA ticket creation."
                " When user diverge from {agent_name} related topics, use {exit_tool}."
                "\nCurrent time: {time}."
            ),
            ("placeholder", "{messages}"),
            ]).partial(agent_name='jira', exit_tool=complete_or_escalate.name)
        ),
    llm=ChatOpenAI(model='gpt-4o-mini'),
    # llm=MockChat(name='worker1',model='worker1'),
    tools=[*shared_tools,
        tools.get_jira_tickets,
        tools.create_jira_ticket,
    ],
    enter_tool=enter_jira_agent,
    exit_tool=complete_or_escalate,
)

@tool
def enter_IT_agent(task:str):
    """Transfers work to 'IT' assistant for handling IT resource solutions and support.
    This agent is capable of:
    - Providing IT resource solutions from an internal database.
    - Raising a new ticket in the FreshService support system.
    - Assisting with troubleshooting common software and hardware issues.
    - Offering guidance on company IT policies and procedures.
    - Helping with network connectivity and access problems.
    - Supporting the setup and configuration of devices and software.
    - Managing user accounts and permissions.
    - Keeping users informed about the status of their requests and any scheduled maintenance.
    """
    return f"The assistant is now the 'IT' assistant with a set of tools to address IT-related tasks."\
            f" The user's intent is unsatisfied. Use the provided tools to complete the task."

IT_agent = Agent(
    name='IT',
    prompt=(
        RunnablePassthrough.assign(**{"time":lambda x:datetime.now()})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a specialised {agent_name} assistant for Virtuos Games Company. "
                " Use the provided tools to assist the user. "
                " Prioritise searching user uploaded content or internal database over the internet. "
                " Please confirm all details before creating a FreshService ticket. You can suggest values based on the conversation."
                " Always ask for confirmation before proceeding with FreshService ticket creation."
                " When user diverge from {agent_name} related topics, use {exit_tool}."
                "\nCurrent time: {time}."
            ),
            ("placeholder", "{messages}"),
            ]).partial(agent_name='IT', exit_tool=complete_or_escalate.name)
        ),
    llm=ChatOpenAI(model='gpt-4o-mini'),
    # llm=MockChat(name='worker1',model='worker1'),
    tools=[*shared_tools,
        tools.search_IT_procedure,
        tools.create_IT_fresh_service_ticket,
    ],
    enter_tool=enter_IT_agent,
    exit_tool=complete_or_escalate,
)



primary_agent = Agent(
    name='primary',
    prompt=(
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
        ),
    llm=ChatOpenAI(model='gpt-4o-mini'),
    # llm=MockChat(),
    tools=[*shared_tools,
        tools.generate_images,
        jira_agent.enter_tool,
        IT_agent.enter_tool,
    ],
    enter_tool=None, # cannot enter, will be entered by default when no more pending tasks
    exit_tool=None, # cannot exit base agent
)
agents = [
    primary_agent, 
    IT_agent, 
    jira_agent,
    ]

added = set()
all_tools = []
for agent in agents:
    for t in agent.tools:
        if t.name not in added:
            added.add(t.name)
            all_tools.append(t)

checkpointer = MemorySaver()
graph = multi_agent_builder(agents).compile(checkpointer=checkpointer, interrupt_before=['human'])
graph.get_graph(
    # xray=True,
    ).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')