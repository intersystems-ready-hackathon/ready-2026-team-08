# LangChain Reference

Use this reference when a Python LangChain app should load models and MCP tools from IRIS by configuration name.

## Prerequisites

- IRIS connection details for the namespace that stores the Config Store entries
- Named LLM and MCP configs already created in IRIS
- Python packages installed for the target providers and MCP support

Pair this skill with `ai-hub-config-store` if the missing step is configuration creation.

## Python Environment Setup

```shell
python -m venv .venv
.venv\Scripts\activate
pip install ./langchain_intersystems-<version>-py3-none-any.whl
pip install mcp langchain-openai langchain-ollama
```

Adjust the wheel filename to match the build you downloaded.

## Required IRIS Config Names

Typical setup:

- `openai`: an `AI.LLM` config with `model_provider`, `model`, and secret-backed `api_key`
- `addition`: an `AI.MCP` config for a stdio MCP server
- `multiplication`: another `AI.MCP` config for a stdio MCP server

The Python side loads by the logical name, not the full dotted identifier.

## Load a Chat Model

```python
import iris
from langchain_intersystems.chat_models import init_chat_model

conn = iris.connect('localhost', 51774, 'USER', '_SYSTEM', 'SYS')
model = init_chat_model('openai', conn)
print(model.invoke('Hello, how are you?'))
```

Use the same logical name you stored in Config Store.

## Load MCP Tools

```python
import asyncio
import iris
from langchain_intersystems import init_mcp_client

conn = iris.connect('localhost', 51774, 'USER', '_SYSTEM', 'SYS')
client = init_mcp_client(['addition', 'multiplication'], conn)
tools = asyncio.run(client.get_tools())
print(tools)
```

Tool enumeration is asynchronous, so use `await` or `asyncio.run()`.

## End-to-End Agent Example

```python
import asyncio
import pprint

import iris
from langchain.agents import create_agent
from langchain_intersystems import init_mcp_client
from langchain_intersystems.chat_models import init_chat_model


async def main() -> None:
	try:
		conn = iris.connect('localhost', 51774, 'USER', '_SYSTEM', 'SYS')
	except Exception as exc:
		raise RuntimeError('Could not connect to IRIS') from exc

	model = init_chat_model('openai', conn)
	client = init_mcp_client(['addition', 'multiplication'], conn)

	try:
		tools = await client.get_tools()
	except Exception as exc:
		raise RuntimeError('Could not load MCP tools from IRIS') from exc

	agent = create_agent(model, tools)

	for question in ['What is 3+5?', 'What is 3x5?']:
		print(question)
		result = await agent.ainvoke({'messages': question})
		pprint.pprint(result)


asyncio.run(main())
```

Use this pattern when IRIS owns the model and MCP configuration, but LangChain owns the Python-side orchestration.

## Practical Rules

- Create or update the named configs in IRIS before running the Python code.
- Keep config names stable so Python callers do not need code changes during rotation.
- Use provider-specific packages such as `langchain-openai` or `langchain-ollama` alongside `langchain-intersystems`.
- Treat MCP tool loading as an async operation that can fail independently of IRIS connection setup.

## Common Failure Modes

- `Connection refused` or login errors from `iris.connect()`: host, port, namespace, or credentials are wrong.
- Model load failure from `init_chat_model()`: the named config does not exist or its provider-specific fields are incomplete.
- Tool load failure from `get_tools()`: the named MCP configs do not exist, or the stdio command cannot start.
- Provider package import errors: install the matching provider integration package in the Python environment.