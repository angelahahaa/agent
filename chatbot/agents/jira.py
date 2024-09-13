

import os
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from chatbot.architecture.clarify import ClarifyState, clarify_agent_builder
from chatbot.tools import (cancel_clarify_request, create_jira_ticket,
                           get_jira_tickets, get_user_info, search_internet)


class JiraState(ClarifyState):
    user_info: str
tools=[get_jira_tickets]
sensitive_tools=[create_jira_ticket, search_internet]
all_tools = tools + sensitive_tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant for Virtuos Games Company. "
                    " Use the provided tools to assist the user. "
                    "\n\nCurrent User:\n<User>\n{user_info}\n</User>"
                    "\nCurrent time: {time}."
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(time=datetime.now())

runnable = (
    prompt
    | llm.bind_tools(tools + sensitive_tools + [cancel_clarify_request])
)

memory = MemorySaver()
builder = StateGraph(JiraState)
builder.add_edge(START, 'init')
builder.add_node('init', lambda state, config: {'user_info':get_user_info.invoke(state, config)})
graph = clarify_agent_builder(
    runnable=runnable,
    tools=tools,
    sensitive_tools=sensitive_tools,
    # node_name_prefix='jira_',
    start='init',
    builder=builder,
    state_schema=ClarifyState,
    ).compile(memory)
graph.get_graph(
    # xray=True,
    ).draw_mermaid_png(output_file_path=f'graphs/{os.path.splitext(os.path.basename(__file__))[0]}.png')
