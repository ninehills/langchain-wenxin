"""Test wenxin API wrapper."""
from typing import List

import pytest
from langchain.callbacks.manager import CallbackManager
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    HumanMessage,
    LLMResult,
)

from langchain_wenxin.chat_models import ChatWenxin
from tests.tools.test_callbacks import FakeCallbackHandler


def test_wenxin_call() -> None:
    """Test valid call to wenxin."""
    chat = ChatWenxin(model="ernie-bot")
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_wenxin_generate() -> None:
    """Test generate method of wenxin."""
    chat = ChatWenxin(model="ernie-bot")
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


def test_wenxin_streaming() -> None:
    """Test streaming tokens from wenxin."""
    chat = ChatWenxin(model="ernie-bot", streaming=True)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_wenxin_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatWenxin(
        model="ernie-bot",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat([message])
    assert callback_handler.llm_streams > 0


@pytest.mark.asyncio
async def test_wenxin_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatWenxin(
        model="ernie-bot",
        streaming=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    chat_messages: List[BaseMessage] = [
        HumanMessage(content="How many toes do dogs have?")
    ]
    result: LLMResult = await chat.agenerate([chat_messages])
    assert callback_handler.llm_streams > 0
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
