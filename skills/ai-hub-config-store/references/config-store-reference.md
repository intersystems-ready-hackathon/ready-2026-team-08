# Config Store Reference

Use this reference to create governed LLM and MCP configuration in IRIS with Wallet-backed secrets.

## What Goes Where

- IRIS Wallet stores secret values such as API keys and bearer tokens.
- Config Store stores structured JSON metadata and `secret://...` references to those secrets.

Keep that split clear. Do not turn the Wallet into a general configuration store.

## Naming Pattern

Use a consistent `area / type / subtype / name` model.

Examples:

- `AI`, `LLM`, `""`, `openai`
- `AI`, `MCP`, `""`, `calculator`
- `AI`, `LLM`, `AWSBedrock`, `prod-west`

The fully qualified dotted name is the same information flattened: `AI.LLM.openai`.

## End-to-End Setup Example

```objectscript
Include %occStatus

Class MyApp.Config.Setup Extends %RegisteredObject
{

ClassMethod ConfigureOpenAIAndMCP(apiKey As %String, pythonPath As %String, serverPath As %String) As %Status
{
    New $NAMESPACE
    Set $NAMESPACE = "%SYS"

    Try {
        // Create security resources for governed access.
        Do ##class(Security.Resources).Create("AIUseResource")
        Do ##class(Security.Resources).Create("AIEditResource")

        // Create a Wallet collection for secrets.
        $$$ThrowOnError(##class(%Wallet.Collection).Create("AISecrets", {
            "UseResource": "AIUseResource",
            "EditResource": "AIEditResource"
        }))

        // Store the OpenAI API key.
        Set secret = {
            "Usage": "CUSTOM",
            "Secret": {"api_key": apiKey}
        }
        $$$ThrowOnError(##class(%Wallet.KeyValue).Create("AISecrets.OpenAI", secret))

        // Create an LLM configuration that references the Wallet secret.
        Set llmConfig = {
            "model_provider": "openai",
            "model": "gpt-4o",
            "api_key": "secret://AISecrets.OpenAI#api_key"
        }
        $$$ThrowOnError(##class(%ConfigStore.Configuration).Create(
            "AI", "LLM", "", "openai", llmConfig
        ))

        // Create an MCP server configuration.
        Set mcpConfig = {
            "command": pythonPath,
            "args": [serverPath],
            "transport": "stdio"
        }
        $$$ThrowOnError(##class(%ConfigStore.Configuration).Create(
            "AI", "MCP", "", "calculator", mcpConfig
        ))

        Quit $$$OK
    } Catch ex {
        Quit ex.AsStatus()
    }
}

}
```

If provisioning may run more than once, add existence checks before calling `Create()` on resources, Wallet collections, and configuration entries.

## Retrieval Pattern

```objectscript
Set llmConfig = ##class(%ConfigStore.Configuration).Get("AI", "LLM", "", "openai")
Set mcpConfig = ##class(%ConfigStore.Configuration).Get("AI", "MCP", "", "calculator")
```

Use the fully qualified dotted name when deletion is clearer:

```objectscript
Do ##class(%ConfigStore.Configuration).Delete("AI.LLM.openai")
Do ##class(%ConfigStore.Configuration).Delete("AI.MCP.calculator")
```

## Common Config Shapes

### LLM Configuration

```objectscript
Set llmConfig = {
    "model_provider": "openai",
    "model": "gpt-4o",
    "api_key": "secret://AISecrets.OpenAI#api_key",
    "temperature": 0.2
}
```

### MCP Configuration

```objectscript
Set mcpConfig = {
    "command": "python",
    "args": ["C:/apps/my-mcp-server.py"],
    "transport": "stdio"
}
```

## Access Control Guidance

- `UseResource`: read or use access for secrets and configs
- `EditResource`: create, modify, or delete access

Be explicit about which IRIS user or role needs each resource. A correctly created config still fails at runtime if the executing user lacks `UseResource`.

## Common Failure Modes

- `Access denied` when resolving a secret: the process user is missing the Wallet collection's `UseResource`.
- `Configuration not found`: area, type, subtype, or name does not match the stored identifier.
- Secret reference does not resolve: the `secret://Collection.Secret#Field` URI points to the wrong collection, key name, or JSON field.
- Raw credentials leak into config JSON: move the secret into the Wallet and keep only the `secret://` reference in Config Store.

## Practical Rules

- Use the Wallet for secret values only.
- Use Config Store for configuration metadata and secret references.
- Keep logical names stable so downstream consumers can load them by name.
- Pair this skill with `ai-hub-langchain` when the consumer is a Python LangChain app.