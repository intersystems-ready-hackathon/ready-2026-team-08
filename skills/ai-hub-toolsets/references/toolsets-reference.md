# ToolSets Reference

Use this reference to turn IRIS logic into a discoverable tool surface.

## Choose the Tool Style

- `%AI.Tool` subclass: best for custom logic, helper methods, or object-shaped parameters
- `%AI.ToolSet` with `<Tool>` elements: best for grouped method-based tools and local XML-defined tools
- `%AI.ToolSet` with `<Query>` elements: best for SQL-first read operations with scalar parameters
- `<Include Class="...">`: reuse tools from another class or ToolSet
- `<MCP>` elements: attach local or remote MCP tools to the same ToolSet

## Method-Based Tool Example

```objectscript
Class MyApp.Tools.Math Extends %AI.Tool
{

Parameter DESCRIPTION = "Perform arithmetic operations used by the billing assistant.";

Method Add(a As %Numeric, b As %Numeric) As %Numeric [ WebMethod ]
{
    Quit a + b
}

Method Multiply(a As %Numeric, b As %Numeric) As %Numeric [ WebMethod ]
{
    Quit a * b
}

}
```

Use a `%AI.Tool` subclass when the tool needs custom logic beyond a simple SQL query.

## Minimal ToolSet Pattern

```objectscript
Class MyApp.ToolSet.BasicMath Extends %AI.ToolSet [ DependsOn = MyApp.Tools.Math ]
{
  XData Definition [ MimeType = application/xml ]
  {
    <ToolSet Name="BasicMath">
      <Description>Basic arithmetic operations with audit logging.</Description>
      <Policies>
        <Audit Class="%AI.Policy.ConsoleAudit" />
      </Policies>
      <Include Class="MyApp.Tools.Math" />
    </ToolSet>
  }
}
```

This is the baseline shape for composing tools and attaching a ToolSet-local audit or authorization policy.

## Inline Tool Pattern

```objectscript
Class MyApp.ToolSet.UtilityTools Extends %AI.ToolSet
{
  XData Definition [ MimeType = application/xml ]
  {
    <ToolSet Name="UtilityTools">
      <Tool Name="Echo" Method="Echo">
        <Description>Echo the input back to the caller.</Description>
      </Tool>
    </ToolSet>
  }

  Method Echo(input As %String) As %String
  {
      Quit "Echo: " _ input
  }
}
```

Use this when the ToolSet class itself implements the tool methods.

## Query-as-Tool Pattern

```objectscript
Class MyApp.ReportTools Extends %AI.ToolSet
{
XData Definition [ MimeType = application/xml ]
{
<ToolSet Name="ReportTools">

  <Query Name="FindOrders"
         Description="Find orders for a customer, optionally filtered by status."
         Arguments="customerId As %Integer, status As %String = ''"
         MaxRows="50">
    <![CDATA[
      SELECT ID, OrderDate, TotalAmount, Status
      FROM MyApp_Orders.Order
      WHERE CustomerID = :customerId
        AND (:status = '' OR Status = :status)
      ORDER BY OrderDate DESC
    ]]>
  </Query>

</ToolSet>
}
}
```

Use `<Query>` when the problem is a SQL-first read operation with scalar parameters.

### JSON Schema Mapping for Query Arguments

| ObjectScript type | JSON Schema type |
|---|---|
| `%Integer`, `%Numeric`, `%Double`, `%Float` | `number` |
| `%Boolean` | `boolean` |
| `%String`, `%Date`, `%Time`, `%TimeStamp` | `string` |

Parameters without a default value are required in the generated schema.

### Common Query Validation Failures

Use compile-time validation to catch mistakes early.

Bad: positional placeholder

```xml
<Query Name="Broken" Arguments="status As %String">
  SELECT ID FROM MyApp.Order WHERE Status = ?
</Query>
```

Fix: use named placeholders such as `:status`.

Bad: SQL placeholder with no matching argument

```xml
<Query Name="Broken" Arguments="customerId As %Integer">
  SELECT ID FROM MyApp.Order WHERE Status = :status
</Query>
```

Fix: every `:param` in the SQL must also appear in `Arguments`.

Bad: optional parameter handled only by default value

```xml
<Query Name="Broken" Arguments="status As %String = ''">
  SELECT ID FROM MyApp.Order WHERE Status = :status
</Query>
```

Fix: guard optional inputs in SQL:

```xml
WHERE (:status = '' OR Status = :status)
```

## Filtering

Use filters to narrow the exposed surface:

- `<Include Class="MyApp.Tools.OrderTools"/>`
- `<Exclude Tool="RawDataDump"/>`
- `<Exclude Match="^Internal"/>`
- `<Exclude Match="^(Internal|Admin|Unsafe)"/>`
- `<Requirement Name="ReadOnly" Value="1"/>` on an include when policies rely on per-tool requirements

Prefer a narrow tool surface up front over adding wide access and trying to claw it back later.

## MCP Composition Inside a ToolSet

```xml
<MCP Name="RemoteCatalog">
  <Remote URL="https://mcp.example.com/mcp"
          AuthType="bearer"
          Token="@{env.MCP_TOKEN}" />
</MCP>
```

Use MCP composition when the agent should see one combined tool surface even though some tools are remote.

Common MCP composition failures:

- bad executable path for local stdio servers
- missing bearer or basic credentials for remote servers
- unreachable remote endpoint at discovery time

Validate local executables, credentials, and network reachability before exposing the ToolSet to users.

## Practical Rules

- Use `%AI.Tool` for richer argument types or procedural logic.
- Use `<Query>` for SQL-heavy read paths with scalar arguments.
- Query results use the standard envelope: `rows`, `row_count`, `truncated`, and `elapsed_ms`.
- Query row caps come from `MaxRows` first, then `QUERYMAXROWS`.
- If a tool needs per-run state, register a specific tool instance instead of only a class.