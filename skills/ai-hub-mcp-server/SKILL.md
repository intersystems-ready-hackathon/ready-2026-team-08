---
name: ai-hub-mcp-server
description: Use when exposing IRIS logic to Claude Desktop or remote MCP clients through iris-mcp-server, %AI.MCP.Service, stdio or HTTP transport, TLS, discovery, namespacing, and two-layer authentication or RBAC.
argument-hint: Describe the MCP deployment target, transport, and security model you need.
---

# AI Hub MCP Server

Use this skill for the full IRIS-to-MCP path: toolsets in IRIS, `%AI.MCP.Service`, `iris-mcp-server`, transport selection, discovery, and security.

## When to Use

- Publishing IRIS tools to Claude Desktop through `stdio`
- Exposing remote MCP endpoints over HTTP or HTTPS
- Figuring out `iris-mcp-server` configuration, discovery, and namespacing
- Troubleshooting tool visibility, authentication failures, or TLS setup

## Core Workflow

1. Build ObjectScript tools or toolsets that represent the IRIS capabilities.
2. Expose them through a `%AI.MCP.Service` subclass.
3. Create the corresponding MCP application in IRIS.
4. Configure `iris-mcp-server` with the right transport and endpoint mapping.
5. Set up both transport-to-IRIS credentials and endpoint-user credentials.
6. Verify discovery, tool namespacing, and security behavior.

## Critical Rules

- Treat `iris-mcp-server` authentication as two separate layers.
- Do not assume `stdio` and HTTP/HTTPS support the same security options.
- Expect service IDs to prefix discovered tool names.
- Pair this skill with `ai-hub-toolsets` for IRIS-side tool design and `ai-hub-policies` for authorization or audit behavior.

## Common Mistakes

- Reusing one credential set for both gateway access and endpoint user identity
- Forgetting that OAuth passthrough is only relevant to HTTP/HTTPS transport
- Missing tool-name prefixes caused by service namespacing
- Debugging discovery issues without checking `iris_status`, logs, or endpoint config first

## References

- [MCP server reference](./references/mcp-server-reference.md)