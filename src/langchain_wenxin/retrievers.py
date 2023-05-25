"""Wrapper around Baidu Wenxin Baizhong knowledge search."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import requests
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


@dataclass
class BaizhongSearchParams:
    # 查询库索引,与项目ID一致
    project_id: str
    # 返回query相关内容数量
    size: int
    # faiss检索返回数量
    db_top: Optional[int] = None
    # 精排排序结果条数
    rank_top: Optional[int] = None
    # 参与精排的最大条数
    rank_size: Optional[int] = None
    # 过滤精排分数低于该值的结果
    doc_score: Optional[float] = None

def para_decode(para: str) -> Tuple[str, dict]:
    """对文心百中返回的 para 字段进行解析的函数"""
    p = json.loads(para)
    return p["content"], {}

class Baizhong(BaseRetriever):
    """Wrapper around Baidu Wenxin Baizhong knowledge search as a retrieval method.

    API docs: https://wenxin.baidu.com/baizhong/doc/?id=Ylaqkc6qb

    How to use:
    ```python
        def para_decode_custom(para: str):
            p = json.loads(para)
            return p["content"], {"index": p.get("index", "")}
        r = Baizhong(endpoint="http://xxxxxxxx:8012", search_params=BaizhongSearchParams(
            project_id="38", size=3, doc_score=0.2),
            para_decode_func=para_decode_custom,
        )
        print(r.get_relevant_documents("xxxxxxxxxxxx"))
    ```
    """
    endpoint: str
    search_params: BaizhongSearchParams
    timeout: int
    para_decode_func = None

    def __init__(self, endpoint: str, search_params: BaizhongSearchParams,
                 para_decode_func = para_decode, timeout: int = 10):
        self.endpoint = endpoint
        self.search_params = search_params
        self.para_decode_func = para_decode_func
        self.timeout = timeout

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_url = f"{self.endpoint}/baizhong/search-api"
        params = {k: v for k, v in asdict(self.search_params).items() if v is not None}
        params["q"] = query
        r = requests.get(search_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        res = r.json()

        if res.get("errorCode") != 0:
            msg = f"Baizhong API returned error: {res.get('errorCode')} - {res.get('errorMsg')}"
            raise RuntimeError(msg)

        docs = []
        for hit in res["hits"]:
            page_content, metadata = self.para_decode_func(hit["_source"]["para"])
            doc = Document(
                page_content=page_content,
                metadata={
                    "_score": hit["_score"],
                    "source": hit["_source"]["title"],
                    "_id": hit["_id"],
                    **metadata,
                },
            )
            docs.append(doc)

        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

