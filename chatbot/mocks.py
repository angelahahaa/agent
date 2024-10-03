import random
import time
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Type, Union)
from uuid import uuid4

from langchain.tools.base import StructuredTool
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, AIMessageChunk
from langchain_core.outputs import (ChatGeneration, ChatGenerationChunk,
                                    ChatResult)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseTool


def mock_tool(name:str, return_direct:bool=False) -> BaseTool:
    if return_direct:
        def fn() -> None:
            return None, {"return_direct": 'image' in name}
        return StructuredTool.from_function(func=fn,name=name,description=name,response_format='content_and_artifact')
    else:
        def fn() -> None:
            return 
        return StructuredTool.from_function(func=fn,name=name,description=name,response_format='content')

def mock_tool_call(name='mock', args={}):
    return {'name': name, 'args': {}, 'id': f'call_{uuid4()}', 'type': 'tool_call'}

class MockChat(BaseChatModel):
    def _llm_type(self) -> str:
        return 'mock'
    
    def __content_tool_calls(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ):
        tools:List[BaseTool] = kwargs.get("tools",[])
        tool_names = {t.name for t in tools}
        if messages:
            if isinstance(messages[-1].content, str):
                txt = messages[-1].content
            elif isinstance(messages[-1].content, List):
                txt = ""
                for content in messages[-1].content:
                    txt += content.get('text',"")
            else:
                raise Exception()
        content = ""
        tool_calls = []
        if txt in tool_names:
            tool_calls = [mock_tool_call(txt)]
        elif tools and random.choice([True, False]):
            tool_calls=[mock_tool_call(random.choice(list(tool_names)))  for _ in range(random.randint(1,3)) ]
        else:
            content=f"you said \"{txt}\""
        return content, tool_calls

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        content, tool_calls = self.__content_tool_calls(messages, **kwargs)
        for c in content:
            time.sleep(0.05)
            yield ChatGenerationChunk(message=AIMessageChunk(content=c))
        yield ChatGenerationChunk(message=AIMessageChunk(content="",tool_calls=tool_calls))
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        content, tool_calls = self.__content_tool_calls(messages, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=tool_calls))]
        )

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return super().bind(tools=tools)
if __name__ == '__main__':
    print(MockChat().bind_tools([mock_tool('a')]).invoke([('human','a')]))