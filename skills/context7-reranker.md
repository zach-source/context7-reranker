# context7-reranker

Enhanced library documentation lookup with query parsing and reranking. Use when user asks about external libraries, frameworks, or packages. Parses natural language queries to extract library names and topics, then retrieves and reranks documentation for maximum relevance.

## When to Use

- User asks "how do I use [library] for [task]"
- User mentions external packages, frameworks, or libraries
- User needs API documentation or code examples
- Questions about library-specific features or patterns

## Workflow

1. **Parse the query** to extract library name and topic:
   ```bash
   context7-reranker parse "How do I use React hooks for state management?" --format context7
   ```

2. **Use extracted parameters** with Context7 MCP:
   - `libraryName`: The canonical library name (e.g., "react")
   - `topic`: The specific topic or feature (e.g., "hooks for state management")

3. **Optionally rerank results** if documentation is too broad:
   ```bash
   echo "$DOCS" | context7-reranker process --query "state management with useState" --top 5
   ```

## Examples

### Parse Query (JSON output)
```bash
context7-reranker parse "FastAPI authentication with JWT"
# {"library_name": "fastapi", "topic": "authentication", "confidence": 0.9}
```

### Parse Query (Context7 format)
```bash
context7-reranker parse "pandas dataframe filtering" --format context7
# libraryName: pandas
# topic: dataframe filtering
```

### Rerank Documentation
```bash
cat docs.md | context7-reranker process --query "middleware setup" --top 3
```

## Environment Variables

Set these for enhanced LLM-based query parsing:

| Variable | Description |
|----------|-------------|
| `LLM_ENDPOINT` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | API key (or use `OPENAI_API_KEY`) |
| `LLM_MODEL` | Model name (default: `gpt-4o-mini`) |

## Integration with Context7 MCP

This skill complements the Context7 MCP server:

1. **context7-reranker**: Parses queries, extracts library/topic, reranks results
2. **Context7 MCP**: Fetches actual documentation from library sources

Typical flow:
```
User query → context7-reranker parse → Context7 resolve-library-id → Context7 get-library-docs → context7-reranker process (rerank)
```
