"""Test Baidu Wenxin API wrapper."""
from typing import Generator

import pytest
from langchain.callbacks.manager import CallbackManager
from langchain.schema import LLMResult

from langchain_wenxin.llms import Wenxin
from tests.tools.test_callbacks import FakeCallbackHandler


def test_wenxin_call() -> None:
    """Test valid call to anthropic."""
    llm = Wenxin(model="ernie-bot")
    output = llm("你好")
    assert isinstance(output, str)


def test_wenxin_streaming() -> None:
    """Test streaming tokens from anthropic."""
    llm = Wenxin(model="ernie-bot")
    generator = llm.stream("你好")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)


def test_wenxin_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = Wenxin(
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm("你好")
    assert callback_handler.llm_streams > 0


@pytest.mark.asyncio
async def test_wenxin_async_generate() -> None:
    """Test async generate."""
    llm = Wenxin()
    output = await llm.agenerate(["你好"])
    assert isinstance(output, LLMResult)


@pytest.mark.asyncio
async def test_wenxin_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = Wenxin(
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["你好"])
    assert callback_handler.llm_streams > 0
    assert isinstance(result, LLMResult)
