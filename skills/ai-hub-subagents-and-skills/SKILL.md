---
name: ai-hub-subagents-and-skills
description: Use when decomposing work with nested agents, %AI.SubAgent, DelegateTask-style delegation, or reusable %AI.Skill definitions that can be exported, loaded, and registered as discoverable tools.
argument-hint: Describe the decomposition pattern, reuse goal, and parent-agent behavior you need.
---

# AI Hub SubAgents and Skills

Use this skill when the problem is decomposition: programmatic sub-agents, nested-agent workflows, or reusable `%AI.Skill` definitions.

## When to Use

- Deciding between `%AI.SubAgent` and `%AI.Skill`
- Implementing delegate-task or recursive-agent patterns
- Exporting or loading skills from external `SKILL.md` sources
- Building a parent agent that hands specialized work to child agents

## Core Workflow

1. Decide whether the child capability should be programmatic, reusable, or externally shareable.
2. Use `%AI.SubAgent` when the parent controls creation directly in code.
3. Use `%AI.Skill` when the capability should be a discoverable tool with metadata and optional tool dependencies.
4. Register the child capability with the parent agent.
5. Keep cost and latency in mind when nesting calls.

## Critical Rules

- `%AI.SubAgent` is the simple programmatic path.
- `%AI.Skill` is the structured, reusable, shareable path.
- Delegate-task tools are an implementation pattern, not the only way to do nested agents.
- Pair this skill with `ai-hub-toolsets` when the child agent needs its own tool surface.

## Common Mistakes

- Treating `%AI.Skill` as just a prompt snippet instead of a tool-registerable sub-agent
- Using nested delegation without a clear reason to split the work
- Ignoring the extra model calls and latency that come with each delegated step
- Assuming external `SKILL.md` loading behaves like a plain local class without validation

## References

- [SubAgents and Skills reference](./references/subagents-and-skills-reference.md)