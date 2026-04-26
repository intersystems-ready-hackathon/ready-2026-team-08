---
name: ai-hub-config-store
description: Use when storing governed LLM or MCP configuration in IRIS Config Store, creating wallet-backed secret:// references, or separating IRIS Wallet secrets from configuration metadata.
argument-hint: Describe the configuration type, secret shape, and consumer you need to support.
---

# AI Hub Config Store

Use this skill when AI Hub components should read model or MCP settings from governed IRIS configuration instead of raw environment variables.

## When to Use

- Creating LLM configurations for ObjectScript, LangChain, or other AI Hub consumers
- Creating MCP server configurations for local or remote tools
- Storing secrets in IRIS Wallet and referencing them from configuration JSON
- Explaining the boundary between Wallet secret storage and Config Store metadata

## Core Workflow

1. Create IRIS security resources for use and edit permissions.
2. Create or choose a Wallet collection.
3. Store secrets with `%Wallet.KeyValue`.
4. Create config entries with `%ConfigStore.Configuration`.
5. Reference Wallet values through `secret://...` URIs.
6. Retrieve configs by area, type, subtype, and name.

## Critical Rules

- Store secrets in the Wallet, not directly in general config JSON when governed storage is expected.
- Use Config Store for structured metadata and references to those secrets.
- Keep naming consistent with the `<area>.<type>.<subtype>.<name>` pattern implied by the docs.
- Pair this skill with `ai-hub-langchain` when the consumer is Python LangChain code.

## Common Mistakes

- Treating Wallet and Config Store as interchangeable
- Forgetting to define `UseResource` and `EditResource`
- Using raw API keys in config JSON when the Wallet should hold them
- Mixing up logical config names with full dotted identifiers

## References

- [Config Store reference](./references/config-store-reference.md)