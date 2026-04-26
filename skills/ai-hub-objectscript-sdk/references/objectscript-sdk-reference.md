# ObjectScript SDK Reference

Use this reference for the core IRIS-native agent workflow: provider setup, reusable agent classes, sessions, streaming, multimodal input, and response handling.

## Core Runtime Objects

- `%AI.Provider`: model-provider abstraction and credential boundary
- `%AI.Agent`: execution engine that coordinates the model, tools, policies, and session history
- `%AI.Agent.Session`: per-conversation history and usage stats
- `%AI.ToolMgr`: tool registry and policy attachment point
- `%AI.LLM.Response`: text, tool, and usage result container

## Minimal Text Assistant

```objectscript
Set provider = ##class(%AI.Provider).Create("openai", {"api_key": apiKey})

Set agent = ##class(%AI.Agent).%New(provider)
Set agent.Model = "gpt-4o"
Set agent.SystemPrompt = "You are a concise operations assistant."

Set session = agent.CreateSession()
Set response = agent.Chat(session, "Summarize today's incidents.")
Write response.Content, !
```

Use this pattern when you need a direct text assistant without subclassing.

## Multi-Turn Session Pattern

```objectscript
Set provider = ##class(%AI.Provider).Create("openai", {"api_key": apiKey})
Set agent = ##class(%AI.Agent).%New(provider)
Set agent.Model = "gpt-4o"
Set agent.SystemPrompt = "You are a support assistant. Keep track of prior turns in the session."

Set session = agent.CreateSession()

Set first = agent.Chat(session, "I'm debugging an order import failure.")
Write "Turn 1: ", first.Content, !

Set second = agent.Chat(session, "Given what I just told you, list the next three checks.")
Write "Turn 2: ", second.Content, !
```

Reuse the same session object across turns. If you create a new session for every call, prior context is lost.

## Reusable Agent Class Pattern

```objectscript
Include (%AI, %occStatus)

Class MyApp.Agent.SupportAssistant Extends %AI.Agent
{

Parameter TOOLSETS = "MyApp.ToolSet.SupportTools";

XData INSTRUCTIONS [ MimeType = "text/markdown" ]
{
You are a support assistant for order-processing operators.
Use tools when they provide a more reliable answer than general knowledge.
}

Method %OnInit() As %Status
{
	If $ISOBJECT(..Provider) Quit $$$OK

	Set apiKey = $SYSTEM.Util.GetEnviron("OPENAI_API_KEY")
	If apiKey = "" {
		Quit $$$ERROR($$$GeneralError, "OPENAI_API_KEY is not set")
	}

	Set ..Provider = ##class(%AI.Provider).Create("openai", {"api_key": apiKey})
	Set ..Model = "gpt-4o"
	Quit $$$OK
}

}
```

Subclass `%AI.Agent` when the behavior should be packaged into a reusable class with default instructions, toolsets, or provider setup.

## Streaming Pattern

```objectscript
Set agent = ##class(MyApp.Agent.SupportAssistant).%New()
Set sc = agent.%Init()
If $$$ISERR(sc) Quit sc

Set session = agent.CreateSession()

Set renderer = ##class(%AI.System.StreamRenderer).%New()
Set renderer.ThinkingPrompt = ""

Set response = agent.StreamChat(
	session,
	"Explain the reconciliation process step by step.",
	renderer,
	"OnChunk"
)

Do renderer.Flush()
Write !, "Final length: ", $LENGTH(response.Content), !
```

Use `StreamChat()` when the caller should see output arrive progressively.

## Multi-Modal Pattern

```objectscript
Set content = []
Do content.%Push({"type": "text", "text": "Describe what is visible in this image."})
Do content.%Push({
	"type": "image_url",
	"image_url": {"url": "https://example.com/inspection.jpg"}
})

Set response = agent.ChatWithContent(session, content)
Write response.Content, !
```

Use content arrays when the request mixes text with image or other structured inputs.

## Response and Stats Pattern

```objectscript
Set response = agent.Chat(session, "Check the current backlog status.")
Write response.Content, !

Set stats = session.GetStats()
Write "Prompt tokens: ", stats."total_prompt_tokens", !
Write "Completion tokens: ", stats."total_completion_tokens", !
Write "Tool calls: ", stats."total_tool_calls", !
```

Read `response.Content` for the user-facing answer and `session.GetStats()` when token or tool usage matters.

## Common Decisions

- Use direct `%AI.Agent` construction for one-off scripts or services.
- Use a subclass when the same prompt, toolset, or provider setup will be reused.
- Use `Chat()` for normal turn-based requests.
- Use `StreamChat()` when the caller needs progressive output.
- Use `ChatWithContent()` when the input is multimodal.

## Pitfalls

- Create the session before multi-turn work, and reuse it for every turn in that conversation.
- Attach tools and toolsets before `CreateSession()` so discovery happens during setup.
- Treat `%AI.Agent.Session` as message history, not a durable application-state container.
- If you need governed credentials instead of raw environment variables, pair this skill with `ai-hub-config-store`.