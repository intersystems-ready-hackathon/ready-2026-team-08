# SubAgents and Skills Reference

Use this reference when the problem is decomposition: child agents, reusable skills, or structured iterative workflows.

## Decision Table

- `%AI.SubAgent`: use when the parent code decides directly that it needs a child agent
- `%AI.Skill`: use when the parent model should discover and invoke a reusable capability by description
- delegate-task tool: use when the parent model itself should decide at runtime when to delegate
- RLM-style loop: use when the work follows a repeated inspect, search, store, and finalize cycle

## Programmatic `%AI.SubAgent` Pattern

```objectscript
Set child = ##class(%AI.SubAgent).Create(
    parentAgent,
    "You are a release-notes specialist.",
    ""
)

Set session = child.CreateSession()
Set response = child.Run(session, "Summarize the breaking changes in this release.")
Write response.Content, !
```

Use this when the parent already knows it wants a child agent and does not need the model to choose that step.

## Delegate-Task Tool Pattern

```objectscript
Class MyApp.Tools.DelegateTask Extends %AI.Tool
{

Property ParentAgent As %AI.Agent;

Method Execute(task As %String, specialistRole As %String = "") As %String [ WebMethod ]
{
    Set systemPrompt = $SELECT(
        specialistRole = "": "You are a helpful assistant.",
        1: "You are a " _ specialistRole _ " assistant."
    )

    Set child = ##class(%AI.SubAgent).Create(..ParentAgent, systemPrompt, "")
    Set session = child.CreateSession()
    Set response = child.Run(session, task)
    Quit response.Content
}

}
```

Use this pattern when delegation should be available as a tool choice during the parent agent's reasoning loop.

## `%AI.Skill` Pattern

```objectscript
Class MyApp.Skill.SummarizeDocument Extends %AI.Skill
{
    Parameter TOOLS = "MyApp.ToolSet.FileTools";

    XData SUMMARY [ MimeType = "text/yaml" ]
    {
name: summarize-document
description: Summarize a document from the filesystem.
parameters:
  - name: userRequest
    description: Path and summarization instructions
    type: string
    required: true
tags:
  - summarization
  - documents
    }

    XData INSTRUCTIONS [ MimeType = "text/markdown" ]
    {
You are a document summarization specialist.
Read the requested file and produce a concise summary under 200 words.
    }
}
```

Use `%AI.Skill` when the capability should be reusable, discoverable, and exportable.

## Register a Skill with a Parent Agent

```objectscript
Set provider = ##class(%AI.Provider).Create("openai", {"api_key": apiKey})
Set agent = ##class(%AI.Agent).%New(provider)
Set agent.Model = "gpt-4o"
Set agent.SystemPrompt = "Use specialized skills when they fit the task better than a direct answer."

Set skill = ##class(MyApp.Skill.SummarizeDocument).%New()
Set skill.ParentAgent = agent
Do agent.ToolManager.AddTool(skill)

Set session = agent.CreateSession()
Set response = agent.Chat(session, "Summarize /data/release-notes.txt")
Write response.Content, !
```

## Export and Load Skills

Export a skill to a directory:

```objectscript
Set skill = ##class(MyApp.Skill.SummarizeDocument).%New()
Set path = skill.ExportSkill("/opt/skills")
Write "Exported to: ", path, !
```

Typical output structure:

```text
/opt/skills/
  summarize-document/
    SKILL.md
```

Load a skill from disk or a Git repository:

```objectscript
Try {
    Set skill = ##class(%AI.Skill).GetSkillFromURI("file:///opt/skills/summarize-document")
} Catch ex {
    Write "Failed to load skill: ", ex.DisplayString(), !
}
```

```objectscript
Try {
    Set skill = ##class(%AI.Skill).GetSkillFromURI("https://github.com/myorg/skills", "summarize-document")
} Catch ex {
    Write "Failed to load skill: ", ex.DisplayString(), !
}
```

Use the filesystem form when the runtime already has a local skill directory. Use the repository form when skills are shared centrally.

## RLM-Style Iterative Loop

Use a structured loop when the child agent should keep working until it gathers enough evidence and explicitly finalizes.

```objectscript
Set session = child.CreateSession()
Set prompt = "Inspect the context, search for evidence, store notes, and finalize only when ready."

For turn=1:1:maxTurns {
    Set response = child.Chat(session, prompt)
    If finalized Quit
    Set prompt = "Continue analysis using tools; call finalize(result) when complete."
}
```

Use this pattern only when the work genuinely benefits from iterative evidence gathering.

## Cost and Complexity Guidance

- Every child agent adds model latency and token usage.
- Delegate only when the child can carry a distinct role, toolset, or reasoning loop.
- Prefer direct `%AI.Agent` execution for simple tasks.
- Prefer `%AI.Skill` when the same specialized capability will be reused across many parents.

## Practical Rules

- `%AI.SubAgent` is the simple programmatic path.
- `%AI.Skill` is the metadata-rich reusable path.
- A delegate-task tool is an orchestration technique built on top of `%AI.SubAgent`.