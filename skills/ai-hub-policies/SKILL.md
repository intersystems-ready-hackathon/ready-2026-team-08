---
name: ai-hub-policies
description: Use when attaching authorization, audit, or discovery policies to ObjectScript tools or ToolSets, sanitizing arguments, enforcing read-only access, or tracing tool execution and denials.
argument-hint: Describe the governance rule, scope, and tool surface you need to protect or observe.
---

# AI Hub Policies

Use this skill for governance over tool execution: allow or deny, sanitize, audit, and compose global or ToolSet-local policies.

## When to Use

- Enforcing read-only or restricted-path behavior
- Logging tool calls, errors, or durations
- Attaching global policies to a `ToolManager`
- Defining ToolSet-local policies in XML

## Core Workflow

1. Decide whether the policy is authorization, audit, or discovery-related.
2. Implement the right base class method.
3. Attach the policy globally or within a ToolSet.
4. Validate the policy against the exact tools and arguments it should govern.
5. Compose global and ToolSet-local rules only when the layering is intentional.

## Critical Rules

- Use `%AI.Policy.Authorization` for allow or deny decisions and argument sanitization.
- Use `%AI.Policy.Audit` for logging and telemetry.
- Use ToolSet-local policies for collection-specific rules and `ToolManager` policies for global rules.
- Authorization policies run global first and ToolSet second; both must allow.
- Audit policies run at both levels when present.
- ToolSet-local XML-configured policies require the XML-enabled pattern from the advanced guide.

## Common Mistakes

- Treating global and ToolSet-local policies as interchangeable
- Forgetting `%XML.Adaptor` and XML property projection for ToolSet-local policy configuration
- Modifying arguments without validating the downstream tool still accepts the sanitized form
- Adding audit policies without deciding what success, failure, and result size should mean operationally

## References

- [Policies reference](./references/policies-reference.md)