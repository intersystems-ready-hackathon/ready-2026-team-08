---
name: ai-hub-rag
description: Use when building IRIS retrieval-augmented generation with %AI.RAG.Embedding, %AI.RAG.VectorStore.IRIS, %AI.RAG.KnowledgeBase, promoted metadata fields, indexing, or ReindexDocument workflows.
argument-hint: Describe the embedding choice, vector store layout, and retrieval behavior you need.
---

# AI Hub RAG

Use this skill for the IRIS-native RAG stack: embeddings, vector store, knowledge base, indexing, and retrieval attachment to agents.

## When to Use

- Building a knowledge base in IRIS
- Choosing between FastEmbed and OpenAI embeddings
- Indexing or reindexing documents
- Exposing retrieval as a tool to an agent
- Using promoted metadata fields for filtered search

## Core Workflow

1. Create an embedding provider.
2. Create a vector store with matching dimensions.
3. Build a `%AI.RAG.KnowledgeBase`.
4. Add documents with stable `source` metadata.
5. Use `ReindexDocument()` when a source document changes.
6. Attach the knowledge base to an agent.

## Critical Rules

- Vector dimensions must match the embedding model.
- `source` metadata matters because it drives deterministic chunk identity and reindex behavior.
- Use `ReindexDocument()` instead of blindly re-adding a changed source.
- Pair this skill with `ai-hub-objectscript-sdk` when the main task is the surrounding agent flow.

## Common Mistakes

- Mixing embeddings and vector-store dimensions
- Ignoring the need for stable source identifiers
- Using `AddDocument()` for updates that should replace earlier chunks
- Skipping promoted metadata fields when filtered retrieval is part of the requirement

## References

- [RAG reference](./references/rag-reference.md)