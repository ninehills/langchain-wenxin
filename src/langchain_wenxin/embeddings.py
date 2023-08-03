"""Wrapper around Wenxin embedding models."""
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from pydantic import BaseModel, Extra, root_validator

from langchain_wenxin.client import WenxinClient


class WenxinEmbeddings(BaseModel, Embeddings):
    """Wrapper around Wenxin embedding models.

    To use, you should have the environment variable ``BAIDU_API_KEY`` and
    ``BAIDU_SECRET_KEY``, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_wenxin.embeddings import WenxinEmbeddings
            wenxin_embed = WenxinEmbeddings(
                model="embedding-v1", baidu_api_key="my-api-key", baidu_secret_key="my-secret-key"
            )
    """

    client: Any = None #: :meta private:
    model: str = "embedding-v1"
    """Model name to use."""

    truncate: Optional[str] = None
    """Truncate embeddings that are too long (> 384 tokens) from start or end ("NONE"|"START"|"END")"""

    baidu_api_key: Optional[str] = None
    """Baidu API key."""

    baidu_secret_key: Optional[str] = None
    """Baidu secret key."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:  # noqa: N805
        """Validate that api key and python package exists in environment."""
        baidu_api_key = get_from_dict_or_env(
            values, "baidu_api_key", "BAIDU_API_KEY"
        )
        baidu_secret_key = get_from_dict_or_env(
            values, "baidu_secret_key", "BAIDU_SECRET_KEY"
        )

        values["client"] = WenxinClient(baidu_api_key=baidu_api_key, baidu_secret_key=baidu_secret_key)

        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Wenxin's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        batch_size = 16
        all_embeddings = []
        for batch in chunks(texts, batch_size):
            output = self.client.embed(
                model=self.model, texts=batch, truncate=self.truncate
            )
            embeddings = output["data"]
            embeddings = sorted(embeddings, key=lambda e: e["index"])
            embeddings = [list(map(float, result["embedding"])) for result in embeddings]
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Wenxin's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        output = self.client.embed(
            model=self.model, texts=[text], truncate=self.truncate
        )
        embedding = output["data"][0]["embedding"]
        return list(map(float, embedding))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
