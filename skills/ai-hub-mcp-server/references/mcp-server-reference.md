# MCP Server Reference

Use this reference to expose IRIS tools to Claude Desktop or remote MCP clients through `iris-mcp-server`.

## Architecture

The MCP flow has five pieces:

1. IRIS tools or ToolSets
2. a `%AI.MCP.Service` subclass that exposes them
3. an IRIS MCP application endpoint such as `/mcp/simple`
4. the `iris-mcp-server` bridge process
5. an MCP client such as Claude Desktop

`iris-mcp-server` handles MCP transport, tool discovery, execution routing, and connection pooling.

## Minimum IRIS Service

```objectscript
Class MyApp.MCP.SimpleService Extends %AI.MCP.Service
{
Parameter SPECIFICATION = "MyApp.ToolSet.Calculator";
}
```

Use this when a ToolSet already exists and only needs an MCP-facing wrapper.

## Claude Desktop over `stdio`

Run the bridge process:

```powershell
iris-mcp-server.exe --config=config.toml run
```

Minimal `config.toml`:

```toml
[mcp]
transport = "stdio"

[[iris]]
name   = "local"
server = { host = "localhost", port = 1972, username = "@{env:WG_USER}", password = "@{env:WG_PASS}" }
pool   = { min = 2, max = 5 }
endpoints = [
	{ path = "/mcp/simple", username = "@{env:APP_USER}", password = "@{env:APP_PASS}" },
]

[logging]
level  = "info"
output = "file"
file   = "C:\\logs\\iris-mcp.log"
```

When using `stdio`, send logs to a file so they do not corrupt the MCP protocol stream.

Claude Desktop configuration:

```json
{
	"mcpServers": {
		"iris": {
			"command": "C:\\tools\\iris-mcp-server.exe",
			"args": [
				"--config=C:\\tools\\config.toml",
				"run"
			]
		}
	}
}
```

## Remote HTTP or HTTPS

Use HTTP for internal networks or a reverse-proxy setup, and HTTPS when the bridge itself should terminate TLS.

```toml
[mcp]
transport = "https"
host      = "0.0.0.0"
port      = 8443

[mcp.tls]
cert = "/etc/certs/server.crt"
key  = "/etc/certs/server.key"

[[iris]]
name   = "production"
server = { host = "iris.example.com", port = 1972, username = "@{env:WG_USER}", password = "@{env:WG_PASS}" }
pool   = { min = 2, max = 10 }
endpoints = [
	{ path = "/mcp/secure", bearer = "@{env:APP_TOKEN}" }
]
tls = { ca_cert = "/etc/certs/iris-ca.crt" }
```

This example uses HTTPS for the client-facing transport and TLS for the `wgproto` connection back to IRIS.

## Two Authentication Layers

Keep these separate:

1. IRIS server authentication: `[[iris]] server.username` and `server.password` for the bridge connection to IRIS
2. MCP endpoint authentication: `endpoints[].username/password/bearer` for the identity presented to each MCP endpoint

Authenticating the bridge connection does not authenticate endpoint requests.

## Discovery and Naming

- Discovery happens through the configured `endpoints` list, or through auto-discovery if endpoints are omitted.
- Tool names are prefixed with a service-derived namespace.
- If tools appear with unexpected names, check the endpoint path and service namespace before assuming discovery failed.

## Secret References

Credential fields support three forms:

- literal strings
- `@{env:VAR}` environment-variable lookups
- `@{vault:path#field}` Vault lookups

Prefer `@{env:...}` or `@{vault:...}` in real deployments.

## Troubleshooting Order

1. Confirm the IRIS MCP application path and `%AI.MCP.Service` dispatch class.
2. Confirm the bridge-level `server.username` and `server.password`.
3. Confirm the per-endpoint `username/password` or `bearer` values.
4. Confirm logging is enabled and check whether tool discovery returned zero tools or failed entirely.
5. Confirm the expected tool namespace and IRIS-side RBAC or policy behavior.

## Common Failure Modes

- Connected to IRIS but every tool call returns `403`: the endpoint credentials are missing or wrong.
- Bridge starts but Claude Desktop sees no tools: the endpoint path, application definition, or service specification is wrong.
- `stdio` transport behaves erratically: logging is going to `stderr` instead of a file.
- HTTPS starts but clients cannot connect: the `mcp.tls` certificate or key path is wrong, or the wrong host and port were configured.

## Practical Rules

- Use `stdio` for Claude Desktop unless you specifically need a remote MCP endpoint.
- Use HTTP or HTTPS when external MCP clients must connect over the network.
- Pair this skill with `ai-hub-toolsets` to design the IRIS-side tool surface and `ai-hub-policies` to enforce authorization or auditing inside those tools.