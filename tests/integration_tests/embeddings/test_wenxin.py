"""Test wenxin embeddings."""
from langchain_wenxin.embeddings import WenxinEmbeddings


def test_wenxin_embedding_documents() -> None:
    """Test wenxin embeddings."""
    documents = ["foo bar"]
    embedding = WenxinEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 384


def test_wenxin_embedding_query() -> None:
    """Test wenxin embeddings."""
    document = "foo bar"
    embedding = WenxinEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) == 384
