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
from langchain_wenxin.llms import Wenxin

# Wenxin model
llm = Wenxin(model="ernie-bot-turbo")
print(llm("你好"))

# stream call
for i in llm.stream("你好"):
    print(i)

# async call
import asyncio
print(asyncio.run(llm._acall("你好")))

# Wenxin chat model
from langchain_wenxin.chat_models import ChatWenxin
from langchain.schema import HumanMessage
llm = ChatWenxin()
print(llm([HumanMessage(content="你好")]))

# Wenxin embeddings model
from langchain_wenxin.embeddings import WenxinEmbeddings
wenxin_embed = WenxinEmbeddings(truncate="END")
print(wenxin_embed.embed_query("hello"))
print(wenxin_embed.embed_documents(["hello"]))
```

Support models:

- ernie-bot: Standard model, <https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11>
    - Also named `wenxin` for compatibility.
- ernie-bot-turbo: Fast model, <https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf>
    - Also named `eb-instant` for compatibility.
- other endpoints: eg: bloomz_7b1 or other custom endpoint.

## Development

```bash
# Create virtual environment
hatch env create
# Activate virtual environment
hatch shell
# Run test
export BAIDU_API_KEY="xxxxxxxx"
export BAIDU_SECRET_KEY="xxxxxxxx"
hatch run test
```

## License

`langchain-wenxin` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
