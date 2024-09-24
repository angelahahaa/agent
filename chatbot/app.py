import dotenv

dotenv.load_dotenv('.env')

import logging
from typing import Annotated, AsyncIterator, Literal, NotRequired, TypeVar
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

from fastapi import FastAPI, HTTPException
from typing import Dict
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, SkipValidation, model_validator, Field, EmailStr
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

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InBlbmdzaGl5YUBhYmMuY29tIn0.jLG70Fquz2t-iFrLbTusjCjcvCRFTju9SV5IX4rwlDE
# schemas
class OpenAIMessage(BaseModel):
    role: Literal['user', 'system', 'tool_deny']
    content: str | List[str | Dict] = Field(..., examples=['hi'])
    tool_call_id: str | None = Field(None, examples=['call_Jja7J89XsjrOLA5r!MEOW!SL'])

    @model_validator(mode='after')
    def check_tool_call_id(self):
        if self.role == 'tool_deny':
            if not self.tool_call_id:
                raise PydanticCustomError(
                    'tool_call_id_missing',
                    'expected tool_call_id',
                )
        return self

    def to_lc_message(self) -> BaseMessage:
        if self.role == 'user':
            return HumanMessage(content=self.content, id=str(uuid4()))
        if self.role == 'system':
            return SystemMessage(content=self.content, id=str(uuid4()))
        if self.role == 'tool_deny':
            # TODO: fill in
            return ToolMessage(content=self.content, id=str(uuid4()))
        raise NotImplementedError()


class InputModel(BaseModel):
    messages: List[OpenAIMessage]


class ConfigurableModel(BaseModel):
    thread_id: str
    email: str | None = Field(None, examples=['angela@abc.com'], description="will be extracted from jwt token if not given.")


class ConfigModel(BaseModel):
    configurable: ConfigurableModel

# fns
bearer_scheme = HTTPBearer(auto_error=False)
def get_config(
    config: ConfigModel, 
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> ConfigModel:
    if credentials:
        token = credentials.credentials
        decoded_token = jwt.decode(token, options={"verify_signature": False})
        email = decoded_token.get("email")
        config.configurable.email = email
    return config.model_dump()

def get_input(
    input: InputModel,
):
    return {"messages":[msg.to_lc_message() for msg in input.messages]}

def to_sse_format(event: str, data: str | None):
    event_str = f"event: {event}\n"
    if data:
        event_str += f"data: {data}\n"
    event_str += "\n"
    return event_str.encode('utf-8')


async def to_streaming_gen(astream_events: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
    async for item in astream_events:
        if item['event'] == "on_chat_model_stream":
            message: AIMessageChunk = item['data']["chunk"]
            if message.content:
                yield to_sse_format(event=item['event'], data=message.json(include={'content'}))
        elif item['event'] == "on_chat_model_end":
            message: AIMessage = item['data']['output']
            yield to_sse_format(event=item['event'],
                                data=message.json(include={'tool_calls'}) if message.tool_calls else None)
        elif item['event'] == "on_tool_end":
            message: ToolMessage = item['data']['output']
            yield to_sse_format(event=item['event'],
                                data=message.json(include={'content', 'artifact', 'tool_call_id', 'name'}))


# api
prefix = "/agent"
app = FastAPI(root_path=prefix)
router = APIRouter(prefix=prefix)

exclude_message_fields = {'additional_kwargs','response_metadata','example','invalid_tool_calls','usage_metadata','tool_call_chunks'}

@router.post("/chat/stream")
async def chat_stream(request: Request, graph_input: Dict = Depends(get_input), graph_config: Dict = Depends(get_config)):
    generator = graph.astream_events(graph_input, graph_config, version='v2')
    return StreamingResponse(to_streaming_gen(generator), media_type="text/event-stream")

@router.post("/chat/invoke")
async def chat_invoke(request: Request, graph_input: Dict = Depends(get_input), graph_config: Dict = Depends(get_config)):
    old_ids = set(msg.id for msg in graph_input['messages'])
    state = graph.invoke(graph_input, graph_config)
    new_messages = []
    is_new_message = False
    for message in state['messages']:
        if message.id in old_ids:
            is_new_message = True
            continue
        if is_new_message:
            new_messages.append(message.dict(exclude=exclude_message_fields))
    return {"messages":new_messages}



app.include_router(router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
