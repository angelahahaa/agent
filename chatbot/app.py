import dotenv

dotenv.load_dotenv('.env')

import logging
from typing import Annotated, AsyncIterator, Literal, NotRequired, Tuple, TypeVar, TypedDict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

logger = logging.getLogger()
import os
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (ConfigurableField, RunnableConfig, RunnablePassthrough)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from chatbot import tools
from chatbot.architecture._multiagent import (AINode, State, create_multiagent_graph)

from fastapi import Body, FastAPI, HTTPException, Path
from typing import Dict
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, SkipValidation, model_validator, Field, EmailStr
from annotated_types import Len
from pydantic_core import PydanticCustomError

from uuid import uuid4
import json
import time
from langchain_core.messages import (AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage,
                                     ToolCallChunk)
from uuid import UUID, uuid4
from chatbot.agents.all_multiagent import graph
from langchain_core.runnables.schema import StreamEvent
import jwt

from langgraph.errors import EmptyInputError

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InBlbmdzaGl5YUBhYmMuY29tIn0.jLG70Fquz2t-iFrLbTusjCjcvCRFTju9SV5IX4rwlDE
MSG_EXCLUDE = {'additional_kwargs','response_metadata','example','invalid_tool_calls','usage_metadata','tool_call_chunks'}

# JSON schemas

class Session(BaseModel):
    session_id:UUID

class ToolDenyMessage(BaseModel):
    type: Literal['tool_deny'] = Field('tool_deny')
    tool_call_id: str = Field(..., examples=['call_Jja7J89XsjrOLA5r!MEOW!SL'])
    content: str = Field(..., examples=['look somewhere else.'])

class ToolDenyInput(BaseModel):
    messages: Annotated[List[ToolDenyMessage], Len(min_length=1)]

class ChatMessage(BaseModel):
    type: Literal['human', 'system']
    content: str | List[str | Dict] = Field(..., examples=['hi'], description="Only for human and system messages")

class ChatInput(BaseModel):
    messages: Annotated[List[ChatMessage], Len(min_length=1)]

class Configurable(BaseModel):
    foo:str|None = ""

class ChatConfig(BaseModel):
    configurable: Configurable = Field(default_factory=Configurable)

session_id_param = Path(..., example=uuid4())


# convert between JSON and python objects
bearer_scheme = HTTPBearer(auto_error=False)

def api_to_lc_message(message:ChatMessage|ToolDenyMessage) -> BaseMessage:
    if isinstance(message, ChatMessage):
        if message.type == 'human':
            return HumanMessage(content=message.content, id=str(uuid4()))
        if message.type == 'system':
            return SystemMessage(content=message.content, id=str(uuid4()))
    if isinstance(message, ToolDenyMessage):
        return ToolMessage(
            name="tool_deny",
            content=f"Tool call denied by user. Reasoning: '{message.content}'. Continue assisting, accounting for the user's input.", 
            tool_call_id=message.tool_call_id,
            id=str(uuid4()), artifact={"reason":message.content})
    raise ValueError(f"Unexpected message: {message}")
    
def get_decoded_jwt(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials:
        token = credentials.credentials
        return jwt.decode(token, options={"verify_signature": False})
    return {}

def get_user_id(decoded_jwt: Dict = Depends(get_decoded_jwt)) -> str:
    # TODO: error if no user_id
    return decoded_jwt.get("preferred_username","")

def get_graph_config(
    config: ChatConfig = Body(...), 
    user_id: str = Depends(get_user_id), 
    session_id: UUID = session_id_param,
) -> Dict:
    config = config.model_dump(by_alias=True)
    config['configurable']['user_id'] = user_id
    config['configurable']['thread_id'] = session_id
    return  config

def get_graph_input(
    input: ChatInput | None = Body(None),
) -> Dict|None:
    return None if input is None else {"messages": [api_to_lc_message(msg) for msg in input.messages]}

def verify_input_for_state(input:Dict|None, config:Dict):
    next_is_tools = "tools" in graph.get_state(config=config).next
    if next_is_tools and input is not None:
        raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"input must be null."
            )
    if not next_is_tools and input is None:
        raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"input must not be null."
            )
def lc_to_api_message(message:BaseMessage, include=None, exclude=MSG_EXCLUDE) -> dict:
    return message.dict(include=include, exclude=exclude)

def sse_event(event:Literal['error']|None=None, data:dict|None=None):
    s = ''
    if event:
        s += f"event: {event}\n"
    if data:
        s += f"data: {json.dumps(data)}\n"
    s += '\n'
    return s.encode('utf-8')

async def lc_to_api_stream_events(astream_events:AsyncIterator[StreamEvent]) -> AsyncIterator[bytes]:
    """convert langgraph asteam events to SSE style stream events.
    For AI message chunks with tool calls, we only stream the content, 
    tool calls will only be streamed as a whole after the full message is generated.
    """
    try:
        async for item in astream_events:
            if item['event'] == "on_chat_model_stream":
                message: AIMessageChunk = item['data']["chunk"]
                if message.content:
                    data = lc_to_api_message(message, include={'content','id'})
                    data.update({'type':'ai'})
                    yield sse_event(data=data)
            elif item['event'] == "on_chat_model_end":
                message: AIMessage = item['data']['output']
                yield sse_event(data=lc_to_api_message(message, exclude=MSG_EXCLUDE | {'content'}))
            elif item['event'] == "on_tool_end":
                message: ToolMessage = item['data']['output']
                yield sse_event(data=lc_to_api_message(message, exclude=MSG_EXCLUDE))
    except EmptyInputError as e:
        yield sse_event(event='error', data={"error":"input must not be null."})
    except Exception as e:
        yield sse_event(event='error', data={"error":f"{type(e).__name__}:{str(e)}"})
# fns

# api
prefix = "/agent"
app = FastAPI(root_path=prefix)
router = APIRouter()


@router.post("/sessions/{session_id}/chat/stream")
async def chat_stream(
    request: Request, 
    input=Depends(get_graph_input),
    config=Depends(get_graph_config),
    ):
    verify_input_for_state(input, config)
    astream_events = graph.astream_events(input, config, version='v2')
    return StreamingResponse(lc_to_api_stream_events(astream_events), media_type="text/event-stream")

@router.post("/sessions/{session_id}/chat/invoke")
async def chat_invoke(
    request: Request, 
    input=Depends(get_graph_input),
    config=Depends(get_graph_config),
    ):
    verify_input_for_state(input, config)
    try:
        state = graph.invoke(input, config)
        messages = [lc_to_api_message(message) for message in state.get('messages',[])]
        return JSONResponse({"messages":messages})
    except EmptyInputError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error":"input must not be null."})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error":f"{type(e).__name__}:{str(e)}"})
    
@router.get("/sessions/{session_id}/chat/history")
def get_chat_history(
    config=Depends(get_graph_config),
):
    snapshot = graph.get_state(config=config)
    messages = [lc_to_api_message(message) for message in snapshot.values.get('messages',[])]
    return {"messages": messages}

@router.post("/sessions/{session_id}/chat/tool-deny")
async def chat_tool_deny(
    request: Request, 
    input:ToolDenyInput=Body(...),
    config=Depends(get_graph_config),
    ):
    values = {"messages": [api_to_lc_message(msg) for msg in input.messages]}
    graph.update_state(config, values, as_node="human")
    return values



@router.post("/sessions")
async def create_session(
    user_id:str=Depends(get_user_id),
):
    session = Session(session_id=uuid4())
    # TODO: add to db
    return session

@router.get("/sessions")
async def get_sessions(
    user_id:str=Depends(get_user_id),
) -> List[Session]:
    ...

@router.post("/sessions/{session_id}/archive")
async def archive_session(
    session_id: UUID=session_id_param,
    user_id:str=Depends(get_user_id),
):
    return {"message":"Session archived"}

# TODO: think about cancel button and history rollback


app.include_router(router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
