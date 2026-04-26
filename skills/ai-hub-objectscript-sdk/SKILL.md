---
name: ai-hub-objectscript-sdk
description: Use when building IRIS-native ObjectScript agents with %AI.Provider, %AI.Agent, sessions, prompts, streaming, response handling, or declarative agent configuration.
argument-hint: Describe the ObjectScript SDK workflow, class design, or runtime behavior you need.
---

# AI Hub ObjectScript SDK

Use this skill for the core IRIS-native agent workflow: provider setup, agent creation, system prompts, sessions, chat APIs, and response handling.

## When to Use

- Building a new `%AI.Agent` or agent subclass
- Choosing between `Chat()`, `StreamChat()`, and content-based requests
- Wiring providers, models, sessions, and toolsets together
- Building reusable ObjectScript classes from the SDK primitives

## Core Workflow

1. Resolve model credentials and build a `%AI.Provider`.
2. Create `%AI.Agent` or subclass `%AI.Agent` for reusable behavior.
3. Set `Model` and `SystemPrompt` early.
4. Attach all tools and toolsets before creating a session.
5. Call `CreateSession()` for any multi-turn workflow.
6. Use the appropriate chat method for text, streaming, or multimodal input.
7. Read `%AI.LLM.Response` content and session stats.

## Critical Rules

- Session context is message history only. It is not a general-purpose state store.
- If you skip `CreateSession()`, repeated `Chat()` calls should be treated as stateless.
- Tool discovery should be treated as part of session setup. Add tools before session creation.
- Pair this skill with `ai-hub-toolsets` when tool design is part of the task.
- Pair this skill with `ai-hub-config-store` when credentials should come from governed IRIS configuration.

## Common Mistakes

- Assuming `Chat()` persists context on its own
- Adding tools after a session already exists
- Treating `%AI.Agent.Session` as arbitrary application state
- Using streaming APIs without deciding how user-visible progress should work
- Reaching for declarative configuration without first understanding the runtime objects it creates

## References

- [ObjectScript SDK reference](./references/objectscript-sdk-reference.md)