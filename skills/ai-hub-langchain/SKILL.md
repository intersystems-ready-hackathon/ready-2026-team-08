---
name: ai-hub-langchain
description: Use when integrating Python LangChain apps with IRIS Config Store through langchain-intersystems, init_chat_model, init_mcp_client, or IRIS-backed model and MCP configuration.
argument-hint: Describe the Python LangChain workflow, IRIS connection, and config names you need.
---

# AI Hub LangChain

Use this skill for the Python side of AI Hub: installing the LangChain integration, connecting to IRIS, and loading governed LLM or MCP config from the Config Store.

## When to Use

- Wiring `langchain-intersystems` into a Python app
- Loading a chat model from IRIS-stored config
- Loading MCP tool definitions from IRIS-stored config
- Turning governed IRIS configuration into a working Python integration

## Core Workflow

1. Create a Python environment and install `langchain-intersystems` plus provider dependencies.
2. Create the underlying IRIS Config Store entries for models and MCP servers.
3. Connect to IRIS with the `iris` Python client.
4. Call `init_chat_model(name, conn)` for model configs.
5. Call `init_mcp_client([...], conn)` for MCP configs.
6. Use the resulting model or tools inside normal LangChain flows.

## Critical Rules

- The Python layer expects the named configs to already exist in IRIS.
- Config creation should be handled in IRIS; model and MCP loading happen in Python.
- Pair this skill with `ai-hub-config-store` whenever the missing piece is how the config should be created or secured.

## Common Mistakes

- Treating the Python snippets as self-contained setup instructions
- Forgetting the `mcp`, `langchain-openai`, or `langchain-ollama` dependencies from the demo
- Using the wrong logical config name when calling `init_chat_model()` or `init_mcp_client()`
- Forgetting that MCP tool enumeration is asynchronous in the Python example flow

## References

- [LangChain reference](./references/langchain-reference.md)