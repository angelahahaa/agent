import asyncio
import json
from typing import AsyncIterator, Dict, Iterator, List, NotRequired, TypedDict

import httpx


class EventStream(TypedDict):
    event:NotRequired[str]
    data:NotRequired[Dict]

def parse_sse(lines:Iterator[str]) -> Iterator[EventStream]:
    event = EventStream()
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            yield event
            event = EventStream()
        elif line.startswith('event: '):
            event['event'] = line[7:]
        elif line.startswith('data: '):
            event['data'] = json.loads(line[6:])
    if event:
        yield event

async def aparse_sse(lines:AsyncIterator[str]) -> AsyncIterator[EventStream]:
    event = EventStream()
    async for line in lines:
        line = line.rstrip('\n')
        if not line:
            yield event
            event = EventStream()
        elif line.startswith('event: '):
            event['event'] = line[7:]
        elif line.startswith('data: '):
            event['data'] = json.loads(line[6:])
    if event:
        yield event
        
class ChatbotAPI:
    def __init__(self, base_url:str, token=None):
        self.base_url = base_url
        self.token = token
        self._client = httpx.Client(timeout=60)
        self._aclient = httpx.AsyncClient(timeout=60)
    @property
    def headers(self):
        return {"Authorization":f"Bearer {self.token}"} if self.token else {}

    def close(self):
        self._client.close()

    async def aclose(self):
        await self._aclient.aclose()

    def get_available_agents(self):
        response = self._client.get(f"{self.base_url}/agents", headers={"Authorization":f"Bearer {self.token}"})
        response.raise_for_status()
        return response.json()

    async def aget_available_agents(self):
        response = await self._aclient.get(f"{self.base_url}/agents", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def create_session(self, agent_name:str):
        response = self._client.post(f"{self.base_url}/sessions", json={'agent_name': agent_name}, headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def acreate_session(self, agent_name:str):
        response = await self._aclient.post(f"{self.base_url}/sessions", json={'agent_name': agent_name}, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_sessions(self) -> List:
        response = self._client.get(f"{self.base_url}/sessions", headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def aget_sessions(self):
        response = await self._aclient.get(f"{self.base_url}/sessions", headers=self.headers)
        response.raise_for_status()
        return response.json()


    def archive_session(self, session_id:str):
        response = self._client.delete(f"{self.base_url}/sessions/{session_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def aarchive_session(self, session_id:str):
        response = await self._aclient.delete(f"{self.base_url}/sessions/{session_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_chat_history(self, session_id:str) -> Dict:
        response = self._client.get(f"{self.base_url}/sessions/{session_id}/chat/history", headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def aget_chat_history(self, session_id:str):
        response = await self._aclient.get(f"{self.base_url}/sessions/{session_id}/chat/history", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def chat_invoke(self, session_id:str, input, config):
        response = self._client.post(f"{self.base_url}/sessions/{session_id}/chat/invoke", json={'input': input, 'config': config}, headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def achat_invoke(self, session_id:str, input, config):
        response = await self._aclient.post(f"{self.base_url}/sessions/{session_id}/chat/invoke", json={'input': input, 'config': config}, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def chat_stream(self, session_id:str, input, config):
        with self._client.stream("POST", f"{self.base_url}/sessions/{session_id}/chat/stream", headers=self.headers, json={"input":input,"config": config}) as response:
            response.raise_for_status()
            for event in parse_sse(response.iter_lines()):
                yield event
    async def achat_stream(self, session_id:str, input, config):
        async with self._aclient.stream("POST", f"{self.base_url}/sessions/{session_id}/chat/stream", headers=self.headers, json={"input":input,"config": config}) as response:
            response.raise_for_status()
            async for event in aparse_sse(response.aiter_lines()):
                yield event

    def get_tools(self, session_id: str) -> List[str]:
        response = self._client.get(f"{self.base_url}/sessions/{session_id}/tools", headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def aget_tools(self, session_id: str) -> List[str]:
        response = await self._aclient.get(f"{self.base_url}/sessions/{session_id}/tools", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def chat_tools_deny(self, session_id: str, input):
        response = self._client.post(f"{self.base_url}/sessions/{session_id}/chat/tools/deny", json=input, headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def achat_tools_deny(self, session_id: str, input, config):
        response = await self._aclient.post(f"{self.base_url}/sessions/{session_id}/chat/tools/deny", json={'input': input, 'config': config}, headers=self.headers)
        response.raise_for_status()
        return response.json()


if __name__ == '__main__':
    chatbot = ChatbotAPI("http://localhost:1234/agent",token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJhbmdlbGFAYWJjLmNvbSIsImV4cCI6MTcyNzM0MjcxNH0.JHO8i-dQLD_ooL3-7AFV3b63m6l7OttK_jzky5dV9HI")
    try:
        print(chatbot.get_available_agents(),end='\n\n')
        print(chatbot.get_sessions(),end='\n\n')
        session = chatbot.create_session('agent-1')
        print(session,end='\n\n')
        session_id = session['session_id']
        print(chatbot.get_chat_history(session_id),end='\n\n')
        for message in chatbot.chat_stream(session_id, {"messages":[{'type':'human','content':'hi'}]}, config={}):
            print(message)
        print()
        print(chatbot.chat_invoke(session_id, {"messages":[{'type':'human','content':'bye'}]}, config={}),end='\n\n')
        print(chatbot.archive_session(session_id),end='\n\n')
    except httpx.HTTPStatusError as e:
        print(e.response.text)
        raise