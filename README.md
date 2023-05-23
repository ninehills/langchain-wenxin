# langchain-wenxin - Langchain Baidu WENXINWORKSHOP wrapper

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-wenxin.svg)](https://pypi.org/project/langchain-wenxin)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langchain-wenxin.svg)](https://pypi.org/project/langchain-wenxin)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install langchain-wenxin
```

## Document

WENXINWORKSHOP API: <https://cloud.baidu.com/doc/WENXINWORKSHOP/s/flfmc9do2>

## How to use

```bash
export BAIDU_API_KEY="xxxxx"                            
export BAIDU_SECRET_KEY="xxxxx"
```

```python3
from langchain_wenxin.llms import Wenxin,ChatWenxin

# Wenxin model
llm = Wenxin(model="eb-instant")
print(llm("你好"))

# Wenxin chat model
from langchain.schema import HumanMessage
llm = ChatWenxin()
print(llm([HumanMessage(content="你好")]))
```

Support models:
- wenxin: 文心一言
- eb-instant: 文心 EB-Lite

## License

`langchain-wenxin` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
