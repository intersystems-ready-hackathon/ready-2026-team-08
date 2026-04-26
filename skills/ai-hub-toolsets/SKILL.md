---
name: ai-hub-toolsets
description: Use when wrapping ObjectScript methods, SQL queries, built-in tools, or external MCP servers into AI-callable ToolSet definitions, including filters, descriptions, JSON schema mapping, and query tools.
argument-hint: Describe the tool surface, input types, and ToolSet behavior you need.
---

# AI Hub ToolSets

Use this skill when the main problem is turning IRIS-side behavior into tools the agent can discover and call.

## When to Use

- Designing method-based tools or XML `ToolSet` definitions
- Exposing SQL queries directly as tools
- Combining local tools, built-ins, policies, and external MCP servers in one ToolSet
- Choosing include or exclude filters and parameter descriptions

## Core Workflow

1. Decide whether the tool should be method-based, query-based, built-in, or MCP-backed.
2. Create a `%AI.Tool` or `%AI.ToolSet` subclass as appropriate.
3. Add descriptions and parameter types the LLM can reason about.
4. Use ToolSet filtering to shape the exposed surface.
5. Attach the ToolSet before session creation.
6. Add policies if the tool surface needs governance.

## Critical Rules

- Prefer `%AI.Tool` subclasses for rich object parameters or custom logic.
- Use inline `<Query>` tools for SQL-first read operations with scalar parameters.
- Use ToolSet-local policies when only that collection of tools needs special governance.
- Pair this skill with `ai-hub-policies` whenever the exposed tools need auth, audit, or sanitization behavior.

## Common Mistakes

- Reaching for `<Query>` when the tool needs complex object parameters
- Forgetting to describe what the tool is for in natural language
- Leaving optional SQL parameters to ObjectScript defaults instead of handling them in SQL
- Treating `Include` and `Exclude` filters as an afterthought after the tool surface is already too broad

## References

- [ToolSets reference](./references/toolsets-reference.md)