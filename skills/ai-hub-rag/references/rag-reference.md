# RAG Reference

Use this reference to build retrieval-augmented generation on IRIS with embeddings, vector storage, and a knowledge-base tool attached to an agent.

## The Stack

Build these in order:

1. `%AI.RAG.Embedding`
2. `%AI.RAG.VectorStore.IRIS`
3. `%AI.RAG.KnowledgeBase`

The knowledge base exposes the retrieval tool the agent calls.

## Embedding Choice

- `%AI.RAG.Embedding.FastEmbed`: local, in-process, no embedding API key, 384 dimensions, model `AllMiniLML6V2`
- `%AI.RAG.Embedding.OpenAI`: provider-backed remote embeddings, 1536 dimensions, model `text-embedding-3-small`

Choose FastEmbed for simple local deployment. Choose OpenAI when remote embedding quality or provider alignment matters more than local simplicity.

## End-to-End Build Pattern

```objectscript
Set emb = ##class(%AI.RAG.Embedding.FastEmbed).Create()

Set vs = ##class(%AI.RAG.VectorStore.IRIS).%New()
Set vs.TableName  = "MyApp.ProductDocs"
Set vs.Dimensions = 384
Set vs.ModelName  = "AllMiniLML6V2"
$$$ThrowOnError(vs.Build())

Set kb = ##class(%AI.RAG.KnowledgeBase).%New()
Set kb.Name = "search_docs"
Set kb.Description = "Product documentation and release notes"
Set kb.TopK = 5
$$$ThrowOnError(kb.Build(emb, vs))
```

Dimensions must match the embedding model exactly.

## Indexing Pattern

```objectscript
Set sc = kb.AddDocument(
	"The reconciliation job supports retry and partial replay.",
	{"source": "ops-guide.txt", "category": "operations"}
)
$$$ThrowOnError(sc)

Set docs = [
	["Release 2026.1 adds batch retry controls.", {"source": "rn-2026.1.txt", "category": "release-notes"}],
	["Webhook delivery is retried up to five times.", {"source": "webhooks.txt", "category": "integration"}]
]
Set chunks = kb.AddDocuments(docs)
Write "Indexed chunks: ", chunks, !
```

Always include a stable `source` value in metadata. It is what makes deterministic chunk IDs and safe reindexing possible.

## Promoted Fields

Use promoted fields when retrieval needs indexed filters such as version, category, or priority.

```objectscript
Set fields = [
	{"name": "version",  "type": "varchar(20)", "description": "Release version such as 2026.1"},
	{"name": "category", "type": "varchar(64)", "description": "Document category"},
	{"name": "priority", "type": "integer",     "description": "Higher values sort as more important"}
]

Set vs = ##class(%AI.RAG.VectorStore.IRIS).%New()
Set vs.TableName  = "MyApp.FilteredDocs"
Set vs.Dimensions = 1536
Set vs.ModelName  = "text-embedding-3-small"
$$$ThrowOnError(vs.Build(fields))
```

Supported field types are `varchar(N)`, `integer`, `float`, and `boolean`.

## OpenAI Embedding Error Handling

```objectscript
Try {
	Set provider = ##class(%AI.Provider).Create("openai", {"api_key": apiKey})
	Set emb = ##class(%AI.RAG.Embedding.OpenAI).Create(provider)
	$$$ThrowOnError(kb.Build(emb, vs))
} Catch ex {
	Write "Embedding setup failed: ", ex.DisplayString(), !
	Quit ex.AsStatus()
}
```

Use this pattern when remote embedding calls may fail because of credentials, network issues, or provider limits.

## Reindex Pattern

```objectscript
Set chunks = kb.ReindexDocument(
	"rn-2026.1.txt",
	"Release 2026.1 adds batch retry controls and operator acknowledgements.",
	{"category": "release-notes", "version": "2026.1"}
)
Write "Reindexed chunks: ", chunks, !
```

Use `ReindexDocument()` when a source changes. Do not call `AddDocument()` for updated content unless you want both the old and new chunks to coexist.

## Attach Retrieval to an Agent

```objectscript
Set provider = ##class(%AI.Provider).Create("openai", {"api_key": apiKey})
Set agent = ##class(%AI.Agent).%New(provider)
Set agent.Model = "gpt-4o"
Set agent.SystemPrompt = "Search the knowledge base before answering product questions."

$$$ThrowOnError(kb.AddToAgent(agent))

Set session = agent.CreateSession()
Set response = agent.Chat(session, "What changed in release 2026.1?")
Write response.Content, !
```

## Debug Checklist

- If `Build()` fails, check whether vector dimensions and model name match the embedding provider.
- If retrieval returns stale information, confirm that updates use `ReindexDocument()`.
- If filtered retrieval is slow or ineffective, confirm that the relevant metadata fields were promoted during `vs.Build(fields)`.
- If a reused vector table rejects the build, check the companion config table for model or dimension mismatch.

## Practical Rules

- `TopK` belongs to the knowledge-base behavior, not the vector-store schema.
- `Build()` creates the main vector table and a companion config table used for mismatch detection.
- Stable `source` metadata is part of the document lifecycle, not an optional label.