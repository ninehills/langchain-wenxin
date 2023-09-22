"""Wrapper around Baidu Wenxin Baizhong knowledge search."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import requests
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever


@dataclass
class BaizhongSearchParams:
    # 查询库索引，与项目 ID 一致
    project_id: int
    # 返回 query 相关内容数量
    size: int
    # faiss 检索返回数量
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
    # 使用离散值过滤的数量，如果设定，则先从百中中搜索出 filter_size 条结果，
    # 再从这些结果中进行离散分布过滤，最终返回不超过 size 条数据
    filter_size: int
    timeout: int
    para_decode_func = None

    def __init__(self, endpoint: str, search_params: BaizhongSearchParams,
                 para_decode_func = para_decode, timeout: int = 10, filter_size: int = 0):
        self.endpoint = endpoint
        self.search_params = search_params
        self.para_decode_func = para_decode_func
        self.timeout = timeout
        self.filter_size = filter_size

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_url = f"{self.endpoint}/baizhong/search-api"
        params = {k: v for k, v in asdict(self.search_params).items() if v is not None}
        params["q"] = query
        if self.filter_size > 0:
            params["size"] = self.filter_size
        r = requests.get(search_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        res = r.json()

        if res.get("errorCode") != 0:
            msg = f"Baizhong API returned error: {res.get('errorCode')} - {res.get('errorMsg')}"
            raise RuntimeError(msg)

        docs = []
        for hit in res["hits"]:
            page_content, metadata = self.para_decode_func(hit["_source"]["para"]) # type: ignore
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

        if self.filter_size > 0:
            # 对结果进行离散分布过滤
            scores = [doc.metadata["_score"] for doc in docs]
            outliers = find_outliers(scores, self.search_params.size)
            return docs[:len(outliers)]
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError


def find_outliers(data: List[float], max_size: int):
    """计算离群值，data 必须是一个倒序排列的数组"""
    if len(data) <= max_size:
        return data
    # 计算相邻元素的差值
    diff = [data[i] - data[i+1] for i in range(len(data) - 1)]
    # 计算平均差值
    avg_diff = np.mean(diff)
    # 离群值列表
    outliers = []
    # 从数组的开始处检查每个元素
    for i in range(len(data) - 1):
        # 如果元素与其后续元素的差值大于平均差值
        if (data[i] - data[i+1]) > avg_diff:
            outliers.append(data[i])
            # 如果找到 3 个离群值则退出循环
            if len(outliers) == max_size:
                break
    # 如果没有找到离群值，返回数组中的最大值作为一个离群值
    if not outliers:
        outliers.append(data[0])

    return outliers
