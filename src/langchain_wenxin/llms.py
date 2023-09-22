"""Wrapper around Baidu Wenxin APIs."""
import logging
import warnings
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM
from langchain.schema.output import GenerationChunk
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, Extra, Field, root_validator

from langchain_wenxin.client import WenxinClient

logger = logging.getLogger(__name__)


class BaiduCommon(BaseModel):
    client: Any = None #: :meta private:
    model_name: str = Field(default="ernie-bot", alias="model")
    """Model name to use. supported models: ernie-bot(wenxin)/ernie-bot-turbo(eb-instant)/other endpoints"""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation. Model default is 0.95.
    range: (0.0, 1.0]."""

    penalty_score: Optional[float] = None
    """Repeating punishment involves penalizing already generated tokens to reduce the occurrence of repetition.
    The larger the value, the greater the punishment. Setting it too high can result in poorer text generation
    for long texts. Model default is 1.0.
    range: [1.0, 2.0]."""

    top_p: Optional[float] = None
    """Diversity influences the diversity of output text.
    The larger the value, the stronger the diversity of the generated text. Model default is 0.8.
    range: (0.0, 1.0]."""

    streaming: bool = False
    """Whether to stream the results."""

    request_timeout: Optional[int] = 600
    """Timeout for requests to Baidu Wenxin Completion API. Default is 600 seconds."""

    baidu_api_key: Optional[str] = None
    """Baidu Cloud API key."""

    baidu_secret_key: Optional[str] = None
    """Baidu Cloud secret key."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:  # noqa: N805
        """Validate that api key and python package exists in environment."""
        baidu_api_key = get_from_dict_or_env(
            values, "baidu_api_key", "BAIDU_API_KEY"
        )
        baidu_secret_key = get_from_dict_or_env(
            values, "baidu_secret_key", "BAIDU_SECRET_KEY"
        )
        values["client"] = WenxinClient(
            baidu_api_key=baidu_api_key,
            baidu_secret_key=baidu_secret_key,
            request_timeout=values["request_timeout"],
        )
        return values


class Wenxin(LLM, BaiduCommon):
    r"""Wrapper around Baidu Wenxin large language models.

    To use, you should have the ``requests`` python package installed, and the
    environment variable ``BAIDU_API_KEY`` and ``BAIDU_SECRET_KEY``, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python
            from langchain_wenxin.llms import Wenxin
            model = Wenxin(model="wenxin", baidu_api_key="my-api-key",
                           baidu_secret_key="my-secret-key")

            # Simplest invocation:
            response = model("What are the biggest risks facing humanity?")
    """

    @root_validator()
    def raise_warning(cls, values: Dict) -> Dict:  # noqa: N805
        """Raise warning that this class is deprecated."""
        warnings.warn(
            "This Wenxin LLM is deprecated. "
            "Please use `from langchain.chat_models import ChatWenxin` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "wenxin-llm"

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

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        r"""Call out to Baidu Wenxin's completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating (not used.).

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "What are the biggest risks facing humanity?"
                response = model(prompt)

        """
        params = {**self._invocation_params, **kwargs}
        if self.streaming:
            stream_resp = self.client.completion_stream(
                model=self.model_name,
                prompt=prompt,
                history=[],
                **params,
            )
            current_completion = ""
            for data in stream_resp:
                result = data["result"]
                if run_manager:
                    run_manager.on_llm_new_token(result, **data)
                current_completion += result
            return current_completion
        response = self.client.completion(
            model=self.model_name,
            prompt=prompt,
            history=[],
            **params,
        )
        return response["result"]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Wenxin's completion endpoint asynchronously."""
        params = {**self._invocation_params, **kwargs}
        if self.streaming:
            stream_resp = self.client.acompletion_stream(
                model=self.model_name,
                prompt=prompt,
                history=[],
                **params,
            )
            current_completion = ""
            async for data in stream_resp:
                delta = data["result"]
                current_completion += delta
                if run_manager:
                    await run_manager.on_llm_new_token(delta, **data)
            return current_completion
        response = await self.client.acompletion(
            model=self.model_name,
            prompt=prompt,
            history=[],
            **params,
        )
        return response["result"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call Wenxin completion_stream and return the resulting generator."""
        params = {**self._invocation_params, **kwargs}

        for token in self.client.completion_stream(
            model=self.model_name,
            prompt=prompt,
            history=[],
            **params):
            yield GenerationChunk(text=token["result"])
            if run_manager:
                run_manager.on_llm_new_token(token["result"])

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,  # noqa: ARG002
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Call Wenxin async completion_stream and return the resulting generator."""
        params = {**self._invocation_params, **kwargs}

        async for token in self.client.acompletion_stream(
            model=self.model_name,
            prompt=prompt,
            history=[],
            **params):
            yield GenerationChunk(text=token["result"])
            if run_manager:
                await run_manager.on_llm_new_token(token["result"])

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens, use text length."""
        return len(text)
