"""Chat wrapper around Baidu Wenxin APIs."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Tuple

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain.schema.output import ChatGenerationChunk
from pydantic import Extra

from langchain_wenxin.llms import BaiduCommon


class ChatWenxin(BaseChatModel, BaiduCommon):
    r"""Wrapper around Baidu Wenxin's large language model.

    To use, you should have the environment variable ``BAIDU_API_KEY`` and
    ``BAIDU_SECRET_KEY``, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python
            from langchain_wenxin.llms import ChatWenxin
            model = ChatWenxin(model="wenxin", baidu_api_key="my-api-key",
                           baidu_secret_key="my-secret-key")

            # Simplest invocation:
            response = model("What are the biggest risks facing humanity?")
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "wenxin-chat"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        d = {}
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.penalty_score is not None:
            d["penalty_score"] = self.penalty_score
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return d

    @property
    def max_message_length(self) -> int:
        """Maximum length of last message."""
        if self.model_name in {"ernie-bot-turbo", "eb-instant"}:
            # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf
            return 11200
        else:
            # https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11
            return 2000

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return {**self._default_params}

    def _convert_messages_to_prompt(
            self, messages: List[BaseMessage]) -> Tuple[str, List[Tuple[str, str]]]:
        """Format a list of messages into prompt and history.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Prompt
            List[Tuple[str, str]]: History
        """
        history = []
        pair: list[Any] = [None, None]
        order_error = "It must be in the order of user, assistant."
        last_message_error = "The last message must be a human message."
        for message in messages[:-1]:
            if message.type == "system":
                history.append((message.content, "OK\n"))
            if pair[0] is None:
                if message.type == "human":
                    pair[0] = message.content
                else:
                    raise ValueError(order_error)
            elif message.type == "ai":
                pair[1] = message.content
                history.append(tuple(pair))
                pair = [None, None]
            else:
                raise ValueError(order_error)
        if not isinstance(messages[-1], HumanMessage):
            raise ValueError(last_message_error)
        return messages[-1].content, history

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt, history = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "history": history, **self._default_params, **kwargs}

        if self.streaming:
            completion = ""
            stream_resp = self.client.completion_stream(**params)
            for delta in stream_resp:
                result = delta["result"]
                completion += result
                if run_manager:
                    run_manager.on_llm_new_token(result)
        else:
            response = self.client.completion(**params)
            completion = response["result"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt, history = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "history": history, **self._default_params, **kwargs}

        if self.streaming:
            completion = ""
            stream_resp = self.client.acompletion_stream(**params)
            async for data in stream_resp:
                delta = data["result"]
                completion += delta
                if run_manager:
                    await run_manager.on_llm_new_token(delta)
        else:
            response = await self.client.acompletion(**params)
            completion = response["result"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt, history = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "history": history, **self._default_params, **kwargs}
        stream_resp = self.client.completion_stream(**params)
        for delta in stream_resp:
            result = delta["result"]
            yield ChatGenerationChunk(message=AIMessageChunk(content=result))
            if run_manager:
                run_manager.on_llm_new_token(result)


    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prompt, history = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "history": history, **self._default_params, **kwargs}
        stream_resp = self.client.acompletion_stream(**params)
        async for data in stream_resp:
            delta = data["result"]
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                await run_manager.on_llm_new_token(delta)

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens, use text length."""
        return len(text)
