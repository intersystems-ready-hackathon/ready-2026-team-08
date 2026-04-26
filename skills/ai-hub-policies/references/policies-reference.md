# Policies Reference

Use this reference to govern tool execution with authorization, audit, and layered policy composition.

## Policy Types

- `%AI.Policy.Authorization`: allow, deny, or sanitize tool calls before execution
- `%AI.Policy.Audit`: record what happened after execution
- `%AI.Policy.Discovery`: advanced extension point for tool visibility decisions

If the requirement is enforcement or logging, start with authorization and audit.

## Authorization Policy Pattern

The key method is:

```objectscript
Method %CanExecute(tool As %String, call As %DynamicObject, metadata As %DynamicObject) As %Status
```

Return `$$$OK` to allow execution or an error status to block it.

### XML-Enabled Path Sanitizer Example

```objectscript
Include (%AI, %occStatus)

Class MyApp.Policy.PathSanitizer Extends (%AI.Policy.Authorization, %XML.Adaptor)
{
  Parameter XMLNAME = "Authorization";

  Property AllowedPath As list Of %String(
    MAXLEN = "",
    XMLITEMNAME = "AllowedPath",
    XMLPROJECTION = "ELEMENT"
  );

  Property Strict As %Boolean(XMLPROJECTION = "ELEMENT");

  Method %CanExecute(tool As %String, call As %DynamicObject, metadata As %DynamicObject) As %Status
  {
    Set path = call.arguments.%Get("path", "")
    If path = "" Quit $$$OK

    // Normalize first so traversal input is checked in canonical form.
    Set normalized = ##class(%File).NormalizeFilename(path)

    Set allowed = 0
    For i=1:1:..AllowedPath.Count() {
      Set prefix = ..AllowedPath.GetAt(i)
      If $E(normalized, 1, $L(prefix)) = prefix {
        Set allowed = 1
        Quit
      }
    }

    If 'allowed && ..Strict {
      Quit $$$ERROR($$$AICoreToolAccessDenied, "Path not in allowed list: " _ normalized)
    }

    Set call.arguments.path = normalized
    Quit $$$OK
  }
}
```

Use this pattern when a ToolSet should only operate under allowed path prefixes.

## Audit Policy Pattern

The key method is:

```objectscript
Method %LogExecution(call, metadata, result, duration, status)
```

Example metrics collector:

```objectscript
Class MyApp.Policy.ToolMetrics Extends %AI.Policy.Audit
{
  Property TotalCalls As %Integer [ InitialExpression = 0 ];
  Property TotalErrors As %Integer [ InitialExpression = 0 ];

  Method %LogExecution(call, metadata, result, duration, status)
  {
    Set ..TotalCalls = ..TotalCalls + 1
    If $$$ISERR(status) {
      Set ..TotalErrors = ..TotalErrors + 1
    }
    Write "Tool=", call.name, " durationMs=", duration, !
  }
}
```

Use audit policies when operational visibility matters more than blocking.

## Global Attachment

Attach policies to the entire tool surface through `ToolManager`:

```objectscript
Do agent.ToolManager.SetAuthPolicy(##class(MyApp.Policy.ReadOnly).%New())
Do agent.ToolManager.SetAuditPolicy(##class(MyApp.Policy.ToolMetrics).%New())
```

## ToolSet-Local Attachment

Attach policies only to one ToolSet when the rule belongs to that collection:

```xml
<Policies>
  <Authorization Class="MyApp.Policy.PathSanitizer">
  <AllowedPath>/tmp</AllowedPath>
  <AllowedPath>/data</AllowedPath>
  <Strict>true</Strict>
  </Authorization>
  <Audit Class="%AI.Policy.ConsoleAudit" />
</Policies>
```

For XML-configured ToolSet-local policies:

- extend the policy base class and `%XML.Adaptor`
- use `XMLPROJECTION = "ELEMENT"` or `"ATTRIBUTE"` on configurable properties
- make sure the XML element names match the property mapping

## Composition Rules

The SDK supports two layers:

1. global policies on `ToolManager`
2. ToolSet-local policies in XML

Composition behavior:

- Authorization: global first, ToolSet second, both must allow
- Audit: both levels run when present
- Discovery: ToolSet-local policy overrides, global policy is the fallback

### Combined Authorization Example

```text
Tool Call: ReadFile(path="/etc/passwd")
  -> Global auth policy allows read-only access
  -> ToolSet path policy checks allowed prefixes
  -> Path policy denies because /etc is not allowed
  -> Final result: denied
```

## Inspecting `metadata` Safely

`metadata` is a runtime-supplied `%DynamicObject`. Its full shape is not exhaustively documented, so treat it defensively.

Useful inspection pattern:

```objectscript
Method %LogExecution(call, metadata, result, duration, status)
{
  If $ISOBJECT(metadata) {
    Write "Metadata: ", metadata.%ToJSON(), !
  }
}
```

Inspect it in a controlled environment first, then write targeted logic for the fields you actually observe.

## Practical Rules

- Sanitize before allowing, not after.
- Keep authorization rules narrow and explicit.
- Keep audit output bounded; do not dump full large payloads unless necessary.
- Validate composed policies with focused runtime tests when the exact outcome matters.