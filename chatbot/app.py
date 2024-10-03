import os
from collections import defaultdict
from datetime import datetime

import aiofiles
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv('.env')

import logging
from typing import Annotated, Any, AsyncIterator, Literal, TypedDict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
import json
from contextlib import asynccontextmanager
from typing import Dict, List
from uuid import UUID, uuid4

import jwt
from annotated_types import Len
from fastapi import (APIRouter, Body, Cookie, Depends, FastAPI, File, HTTPException, Path, Query, Request, Security, UploadFile,
                     status)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough)
from langchain_core.runnables.schema import StreamEvent
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.checkpoint.postgres.aio import (AsyncConnectionPool, AsyncPostgresSaver)
from langgraph.errors import EmptyInputError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from pydantic import (AfterValidator, AnyHttpUrl, AnyUrl, BaseModel, Field, SkipValidation, model_validator)
from typing_extensions import TypedDict

from chatbot import config
from chatbot.agents.all_multiagent import (IT_agent, get_all_tools, jira_agent, primary_agent)
from chatbot.architecture.base import PartialToolNode, return_direct_condition, no_tools_agent_builder
from chatbot.architecture.multiagent import multi_agent_builder
from chatbot.database.session_db import SessionInfo, SessionManager
from chatbot.mocks import MockChat, mock_tool
from fastapi.middleware.cors import CORSMiddleware

MSG_EXCLUDE = {
    'additional_kwargs', 'response_metadata', 'example', 'invalid_tool_calls', 'usage_metadata', 'tool_call_chunks'
}

# JSON schemas


class CreateSessionRequest(BaseModel):
    agent_name: str = Field(..., examples=['mock'])


class ToolDenyMessage(BaseModel):
    type: Literal['tool'] = Field('tool')
    tool_call_id: str = Field(..., examples=['call_Jja7J89XsjrOLA5r!MEOW!SL'])
    reason: str = Field(..., examples=['look somewhere else.'])


class ToolDenyInput(BaseModel):
    messages: Annotated[List[ToolDenyMessage], Len(min_length=1)]


class ChatMessage(BaseModel):
    type: Literal['human', 'system', 'tool']
    content: str | List[str | Dict] = Field(..., examples=['hi'])


class ChatInput(BaseModel):
    messages: Annotated[List[ChatMessage], Len(min_length=1)]


class Configurable(BaseModel):
    foo: str | None = ""
    data_ids: List[str] = Field(default_factory=list)


class ChatConfig(BaseModel):
    configurable: Configurable = Field(default_factory=Configurable)


SessionId = Annotated[UUID, Path(..., examples=[uuid4()])]
bearer_scheme = HTTPBearer(auto_error=False)

# convert between JSON and python objects


def api_to_lc_message(message: ChatMessage | ToolDenyMessage) -> BaseMessage:
    if isinstance(message, ChatMessage):
        if message.type == 'human':
            return HumanMessage(content=message.content, id=str(uuid4()))
        if message.type == 'system':
            return SystemMessage(content=message.content, id=str(uuid4()))
    if isinstance(message, ToolDenyMessage):
        return ToolMessage(
            name="user_deny",
            content=
            f"Tool call denied by user. Reasoning: '{message.reason}'. Continue assisting, accounting for the user's input.",
            tool_call_id=message.tool_call_id,
            id=str(uuid4()),
            artifact={"reason": message.reason})
    raise ValueError(f"Unexpected message: {message}")


def get_decoded_jwt(
        bearer_token: HTTPAuthorizationCredentials|None = Security(bearer_scheme),
        param_token: str = Query(None, alias="token")
        ):
    # Prioritize header-based authentication
    if bearer_token:
        token = bearer_token.credentials
    # Fall back to cookie-based authentication if no header is provided
    elif param_token:
        token = param_token
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    
    return jwt.decode(token, options={"verify_signature": False})

DecodedJwt = Annotated[Dict, Depends(get_decoded_jwt)]


def get_username(decoded_jwt: DecodedJwt) -> str:
    username = decoded_jwt.get("preferred_username")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="preferred_username not found in jwt")
    return username


Username = Annotated[str, Depends(get_username)]


def get_graph_input(input: ChatInput | None = Body(None),) -> Dict | None:
    return None if input is None else {"messages": [api_to_lc_message(msg) for msg in input.messages]}


GraphInput = Annotated[Dict | None, Depends(get_graph_input)]


# fns
async def verify_input_for_state(agent: CompiledStateGraph, input: Dict | None, config: RunnableConfig):
    snapshot = await agent.aget_state(config=config)
    has_next = bool(snapshot.next)
    if has_next and input is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"input must be null.")
    if not has_next and input is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"input must not be null.")


def lc_to_api_message(message: BaseMessage, include=None, exclude=MSG_EXCLUDE) -> dict:
    return message.dict(include=include, exclude=exclude)


def sse_event(data: dict | str, event: str | None = None):
    """
    Create a Server-Sent Event string from an event and data.
    
    :param event: The name of the event. Can be None.
    :param data: The data associated with the event. Can be None.
    :return: A formatted Server-Sent Event string.
    """
    if isinstance(data, dict):
        data = json.dumps(data)
    lines = []
    if event is not None:
        lines.append(f"event: {event}")
    # Data must be split into lines of 2048 characters or less
    for i in range(0, len(data), 2048):
        lines.append(f"data: {data[i:i+2048]}")
    s = "\n".join(lines) + "\n\n"
    return s.encode()

# api
class Agent(TypedDict):
    graph: CompiledStateGraph
    tools: List[BaseTool]


async def get_agent(request: Request, session_id: SessionId):
    session_manager: SessionManager = request.app.state.session_manager
    session_info = await session_manager.get_session(session_id)
    if not session_info:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session_id")
    agents: Dict[str, Agent] = request.app.state.agents
    return agents[session_info.agent_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    # initialise session manager
    session_manager = SessionManager(config.CONN_STRING)
    await session_manager.setup()
    app.state.session_manager = session_manager
    # initialise agents
    agents: Dict[str, Agent] = dict()
    async with AsyncConnectionPool(
            conninfo=config.CONN_STRING,
            max_size=20,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            },
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)  # type: ignore
        await checkpointer.setup()
        # agents with no tools
        agents[f'gpt-4o-mini'] = Agent(
            graph=no_tools_agent_builder(ChatOpenAI(model='gpt-4o-mini')).compile(checkpointer=checkpointer, interrupt_before=['agent']),
            tools=[],
        )
        agents[f'gpt-4o'] = Agent(
            graph=no_tools_agent_builder(ChatOpenAI(model='gpt-4o')).compile(checkpointer=checkpointer, interrupt_before=['agent']),
            tools=[],
        )
        agents[f'mock'] = Agent(
            graph=no_tools_agent_builder(MockChat()).compile(checkpointer=checkpointer, interrupt_before=['agent']),
            tools=[],
        )
        # preset1
        assistants = [primary_agent, jira_agent, IT_agent]
        builder = multi_agent_builder(assistants)
        agents['agent-1'] = Agent(
            graph=builder.compile(checkpointer=checkpointer, interrupt_before=['tools'] + [assistant.name for assistant in assistants]),
            tools=get_all_tools(assistants),
        )
        # add agents to app state
        app.state.agents = agents
        yield

    # Clean up
    await session_manager.pool.close()


prefix = "/agent"
app = FastAPI(root_path=prefix)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(lifespan=lifespan)


@router.get("/agents")
def get_available_agents(request: Request):
    return list(request.app.state.agents.keys())


@router.post("/sessions/{session_id}/chat")
async def chat(
    request: Request,
    username: Username,
    session_id: SessionId,
    config: ChatConfig = Body(default_factory=ChatConfig),
    input: ChatInput = Body(...),
):
    agent = await get_agent(request, session_id)
    graph = agent['graph']


    config_dict = config.model_dump()
    config_dict['configurable']['username'] = username
    config_dict['configurable']['thread_id'] = str(session_id)
    graph_config = RunnableConfig(**config_dict)

    # snapshot = await graph.aget_state(config=graph_config)
    # if snapshot.next:
    #     return JSONResponse(
    #         status_code=status.HTTP_409_CONFLICT, 
    #         content={"error":"The session is in a state where new chat content cannot be posted. Please use the GET /sessions/{session_id}/chat/stream or /sessions/{session_id}/chat/invoke endpoints to receive updates."})

    graph_input = {"messages": [api_to_lc_message(msg) for msg in input.messages]}
    session_manager: SessionManager = app.state.session_manager
    await session_manager.update_session(session_id)

    await graph.ainvoke(graph_input, graph_config)
    return JSONResponse({"status": "ok"})

@router.get("/sessions/{session_id}/chat/stream")
async def chat_stream(
    request: Request,
    username: Username,
    session_id: SessionId,
):
    agent = await get_agent(request, session_id)
    graph = agent['graph']

    graph_config = RunnableConfig(configurable={"username":username, "thread_id":str(session_id)})

    snapshot = await graph.aget_state(config=graph_config)
    if not snapshot.next:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT, 
            content={"error":"The session is in a state to receive updates. Please use the POST /sessions/{session_id}/chat to post new chat content."})


    async def generator() -> AsyncIterator[bytes]:
        """convert langgraph asteam events to SSE style stream events.
        For AI message chunks with tool calls, we only stream the content,
        tool calls will only be streamed as a whole after the full message is generated.
        """
        disconnected = False
        ai_message:AIMessageChunk | None = None
        try:
            logging.info('start')
            async for item in graph.astream_events(None, graph_config, version='v2'):
                disconnected = await request.is_disconnected()
                if disconnected:
                    logging.info('disconnected')
                    break
                if item['event'] == 'on_chain_end' and item['name'] == "_write":
                    # ai message written to checkpoint, can flush
                    ai_message = None
                elif item['event'] == "on_chat_model_start":
                    ai_message = AIMessageChunk(content='')
                elif item['event'] == "on_chat_model_stream":
                    ai_message_chunk: AIMessageChunk = item['data']["chunk"]  # type: ignore
                    ai_message += ai_message_chunk # type: ignore
                    # yield content if any
                    if ai_message_chunk.content:
                        data = lc_to_api_message(ai_message_chunk, include={'content', 'id'})
                        data.update({'type': 'ai'})
                        yield sse_event(data=data)
                elif item['event'] == "on_chat_model_end":
                    # yield tool calls if any
                    if ai_message and ai_message.tool_calls:
                        data = lc_to_api_message(ai_message_chunk, exclude=MSG_EXCLUDE | {'content'})
                        data.update({'type': 'ai'})
                        yield sse_event(data=data)
                elif item['event'] == "on_tool_start":
                    kwargs:Dict = item['data']['input']   # type: ignore
                    yield sse_event(event='tool-start',data=kwargs)
                elif item['event'] == "on_tool_end":
                    tool_message: ToolMessage = item['data']['output']  # type: ignore
                    yield sse_event(data=lc_to_api_message(tool_message, exclude=MSG_EXCLUDE))
            if not disconnected:
                yield sse_event(event='stream-end', data={})
        except asyncio.CancelledError as e:
            raise
        except Exception as e:
            logger.error(e)
            yield sse_event(event='error', data={"error": f"{type(e).__name__}:{str(e)}"})
        finally:
            async def cleanup():
                logger.warning(f"stream cancelled, cleaning up...")
                await graph.aupdate_state(graph_config, {"messages":[ai_message]})
            asyncio.shield(cleanup())

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/sessions/{session_id}/chat/invoke")
async def chat_invoke(
    request: Request,
    username: Username,
    session_id: SessionId,
):
    agent = await get_agent(request, session_id)
    graph = agent['graph']

    graph_config = RunnableConfig(configurable={"username":username, "thread_id":str(session_id)})

    snapshot = await graph.aget_state(config=graph_config)
    if not snapshot.next:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT, 
            content={"error":"The session is in a state to receive updates. Please use the POST /sessions/{session_id}/chat to post new chat content."})

    try:
        updates:List[Dict[str, Any]] = await graph.ainvoke(input, graph_config, stream_mode='updates')
        messages = []
        for update in updates:
            for _, state in update.items():
                messages.extend([lc_to_api_message(message) for message in state.get("messages", [])])
        return JSONResponse({"messages": messages})
    except Exception as e:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"error": f"{type(e).__name__}:{str(e)}"})


@router.get("/sessions/{session_id}/chat/history")
async def get_chat_history(
    request: Request,
    session_id: SessionId,
):
    agent = await get_agent(request, session_id)
    snapshot = await agent['graph'].aget_state(config=RunnableConfig(configurable={'thread_id': str(session_id)}))
    messages = [lc_to_api_message(message) for message in snapshot.values.get('messages', [])]
    return {"messages": messages}


@router.get("/sessions/{session_id}/tools")
async def tools(
    request: Request,
    session_id: SessionId,
):
    agent = await get_agent(request, session_id)
    return [t.name for t in agent['tools']]


@router.post("/sessions/{session_id}/chat/tools/deny")
async def chat_tools_deny(
    request: Request,
    input: Annotated[ToolDenyInput, Body(...)],
    session_id: SessionId,
):
    agent = await get_agent(request, session_id)
    graph = agent['graph']
    values = {"messages": [api_to_lc_message(msg) for msg in input.messages]}
    await graph.aupdate_state(RunnableConfig(configurable={'thread_id': str(session_id)}), values, as_node="human")
    return values


@router.post("/sessions")
async def create_session(
    request: Request,
    username: Username,
    body: CreateSessionRequest,
) -> SessionInfo:
    session_manager: SessionManager = app.state.session_manager
    agents: Dict[str, Agent] = request.app.state.agents
    if not body.agent_name in agents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid agent_name, must be one of: {list(agents.keys())}")
    session_info = await session_manager.create_session(username, body.agent_name)
    return session_info


@router.get("/sessions")
async def get_sessions(username: Username) -> List[SessionInfo]:
    session_manager: SessionManager = app.state.session_manager
    session_infos = await session_manager.get_sessions(username)
    return session_infos


@router.delete("/sessions/{session_id}")
async def archive_session(session_id: SessionId):
    session_manager: SessionManager = app.state.session_manager
    await session_manager.archive_session(session_id)
    return {"message": "Session archived successfully."}


from chatbot.database import data_db


@router.post("/data")
async def upload(
        data_type: Literal['file', 'web', 'github'] = Body(...),
        url: str | None = Body(None),
        file: UploadFile | None = File(None),
):
    """ Uploads data to vector store
    """
    data_id = f"data_{uuid4()}"
    if data_type == 'file':
        if file is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"data_type '{data_type}' requires upload file.")
        ext = os.path.split(file.filename)[1] if file.filename else ""
        path = os.path.join(config.TEMP_DIR, data_id + ext)
        async with aiofiles.open(path, 'wb') as async_temp_file:
            chunk_size = 1024 * 1024  # 1 MB, read in chunks to avoid RAM overload
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                await async_temp_file.write(chunk)
        asyncio.create_task(data_db.put_file(data_id, path, metadata={'source': file.filename or 'unkown'}))
    elif data_type == 'web':
        if url is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"data_type '{data_type}' requires url.")
        asyncio.create_task(data_db.put_web(data_id, url))

    else:
        raise NotImplementedError()
    return {'data_id': data_id}


@app.get("/data/{data_id}/status")
def check_data_status(data_id: str):
    ...


@router.post("/sessions/{session_id}/data/{data_id}")
def add_data_to_session(session_id: SessionId, data_id: str):
    """ Does not verify if the data_id exists or is completely uploaded. 
    This piece of data will be ignored if data_id is invalid
    """
    ...


# TODO: think about cancel button and history rollback

app.include_router(router)
if __name__ == "__main__":
    import asyncio
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
