import dotenv

dotenv.load_dotenv('.env')

import logging
from typing import (Annotated, AsyncIterator, Literal)
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger()
import json
from contextlib import asynccontextmanager
from typing import Dict, List
from uuid import UUID, uuid4

import jwt
from annotated_types import Len
from fastapi import (APIRouter, Body, Depends, FastAPI, HTTPException,
                     Path, Security, status)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import (AIMessage, AIMessageChunk, BaseMessage,
                                     HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.runnables import (RunnableConfig)
from langchain_core.runnables.schema import StreamEvent
from langgraph.errors import EmptyInputError
from pydantic import (BaseModel, Field)

from chatbot import config
from chatbot.agents.all_multiagent import graph
from chatbot.database.session_db import SessionManager, SessionInfo

MSG_EXCLUDE = {
    'additional_kwargs', 'response_metadata', 'example', 'invalid_tool_calls', 'usage_metadata', 'tool_call_chunks'
}

# JSON schemas

class ToolDenyMessage(BaseModel):
    type: Literal['tool_deny'] = Field('tool_deny')
    tool_call_id: str = Field(..., examples=['call_Jja7J89XsjrOLA5r!MEOW!SL'])
    content: str = Field(..., examples=['look somewhere else.'])


class ToolDenyInput(BaseModel):
    messages: Annotated[List[ToolDenyMessage], Len(min_length=1)]


class ChatMessage(BaseModel):
    type: Literal['human', 'system', 'tool']
    content: str | List[str | Dict] = Field(..., examples=['hi'])

class ChatInput(BaseModel):
    messages: Annotated[List[ChatMessage], Len(min_length=1)]


class Configurable(BaseModel):
    foo: str | None = ""


class ChatConfig(BaseModel):
    configurable: Configurable = Field(default_factory=Configurable)


SessionId = Annotated[UUID, Path(..., example=uuid4())]
bearer_scheme = HTTPBearer()

# convert between JSON and python objects


def api_to_lc_message(message: ChatMessage | ToolDenyMessage) -> BaseMessage:
    if isinstance(message, ChatMessage):
        if message.type == 'human':
            return HumanMessage(content=message.content, id=str(uuid4()))
        if message.type == 'system':
            return SystemMessage(content=message.content, id=str(uuid4()))
    if isinstance(message, ToolDenyMessage):
        return ToolMessage(
            name="tool_deny",
            content=
            f"Tool call denied by user. Reasoning: '{message.content}'. Continue assisting, accounting for the user's input.",
            tool_call_id=message.tool_call_id,
            id=str(uuid4()),
            artifact={"reason": message.content})
    raise ValueError(f"Unexpected message: {message}")


def get_decoded_jwt(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials:
        token = credentials.credentials
        return jwt.decode(token, options={"verify_signature": False})
    return {}


DecodedJwt = Annotated[Dict, Depends(get_decoded_jwt)]


def get_username(decoded_jwt: DecodedJwt) -> str:
    username = decoded_jwt.get("preferred_username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="preferred_username not found in jwt"
        )
    return username


Username = Annotated[str, Depends(get_username)]


def get_graph_config(
    config: Annotated[ChatConfig, Body(...)],
    username: Username,
    session_id: SessionId,
) -> RunnableConfig:
    graph_config = config.model_dump(by_alias=True)
    graph_config['configurable']['username'] = username
    graph_config['configurable']['thread_id'] = session_id
    return RunnableConfig(**graph_config)


GraphConfig = Annotated[RunnableConfig, get_graph_config]


def get_graph_input(input: ChatInput | None = Body(None),) -> Dict | None:
    return None if input is None else {"messages": [api_to_lc_message(msg) for msg in input.messages]}


GraphInput = Annotated[Dict | None, get_graph_input]

# fns


def verify_input_for_state(input: Dict | None, config: RunnableConfig):
    next_is_tools = "tools" in graph.get_state(config=config).next
    if next_is_tools and input is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"input must be null.")
    if not next_is_tools and input is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"input must not be null.")


def lc_to_api_message(message: BaseMessage, include=None, exclude=MSG_EXCLUDE) -> dict:
    return message.dict(include=include, exclude=exclude)


def sse_event(event: Literal['error'] | None = None, data: dict | None = None):
    s = ''
    if event:
        s += f"event: {event}\n"
    if data:
        s += f"data: {json.dumps(data)}\n"
    s += '\n'
    return s.encode('utf-8')


async def lc_to_api_stream_events(astream_events: AsyncIterator[StreamEvent]) -> AsyncIterator[bytes]:
    """convert langgraph asteam events to SSE style stream events.
    For AI message chunks with tool calls, we only stream the content, 
    tool calls will only be streamed as a whole after the full message is generated.
    """
    try:
        async for item in astream_events:
            if item['event'] == "on_chat_model_stream":
                message: AIMessageChunk = item['data']["chunk"]  # type: ignore
                if message.content:
                    data = lc_to_api_message(message, include={'content', 'id'})
                    data.update({'type': 'ai'})
                    yield sse_event(data=data)
            elif item['event'] == "on_chat_model_end":
                message: AIMessage = item['data']['output']  # type: ignore
                yield sse_event(data=lc_to_api_message(message, exclude=MSG_EXCLUDE | {'content'}))
            elif item['event'] == "on_tool_end":
                message: ToolMessage = item['data']['output']  # type: ignore
                yield sse_event(data=lc_to_api_message(message, exclude=MSG_EXCLUDE))
    except EmptyInputError as e:
        yield sse_event(event='error', data={"error": "input must not be null."})
    except Exception as e:
        yield sse_event(event='error', data={"error": f"{type(e).__name__}:{str(e)}"})


# api
session_manager = SessionManager()
@asynccontextmanager
async def lifespan(app: FastAPI):
    await session_manager.create_pool(
        config.DB_USER, 
        config.DB_PASSWORD, 
        config.DB_DATABASE, 
        config.DB_HOST)
    await session_manager.create_table_if_not_exists()
    yield
    # Clean up the ML models and release the resources
    await session_manager.pool.close()

prefix = "/agent"
app = FastAPI(root_path=prefix)
router = APIRouter(lifespan=lifespan)


@router.post("/sessions/{session_id}/chat/stream")
async def chat_stream(
    input: GraphInput,
    config: GraphConfig,
):
    verify_input_for_state(input, config)
    astream_events = graph.astream_events(input, config, version='v2')
    return StreamingResponse(lc_to_api_stream_events(astream_events), media_type="text/event-stream")


@router.post("/sessions/{session_id}/chat/invoke")
async def chat_invoke(
    input: GraphInput,
    config: GraphConfig,
):
    verify_input_for_state(input, config)
    try:
        state = graph.invoke(input, config)
        messages = [lc_to_api_message(message) for message in state.get('messages', [])]
        return JSONResponse({"messages": messages})
    except EmptyInputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": "input must not be null."})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": f"{type(e).__name__}:{str(e)}"})


@router.get("/sessions/{session_id}/chat/history")
def get_chat_history(config: GraphConfig):
    snapshot = graph.get_state(config=config)
    messages = [lc_to_api_message(message) for message in snapshot.values.get('messages', [])]
    return {"messages": messages}


@router.post("/sessions/{session_id}/chat/tool-deny")
async def chat_tool_deny(
    input: Annotated[ToolDenyInput, Body(...)],
    config: GraphConfig,
):
    values = {"messages": [api_to_lc_message(msg) for msg in input.messages]}
    graph.update_state(config, values, as_node="human")
    return values


@router.post("/sessions")
async def create_session(username: Username) -> SessionInfo:
    session_info = await session_manager.create_session(username)
    return session_info


@router.get("/sessions")
async def get_sessions(username: Username) -> List[SessionInfo]:
    session_infos = await session_manager.get_sessions(username)
    return session_infos


@router.delete("/sessions/{session_id}")
async def archive_session(session_id: SessionId):
    await session_manager.archive_session(session_id)
    return {
        "message": "Session archived successfully."
    }


# TODO: think about cancel button and history rollback

app.include_router(router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
