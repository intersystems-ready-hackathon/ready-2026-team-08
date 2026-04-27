# TEAM 08 — IRIS CHAT EDI

## Project Summary

An agentic chatbot that lets ops teams query IRIS event logs (`Ens_Util.Log`) in plain English. A langchain ReAct agent picks the right MCP tool — including a custom `query_iris_logs` tool — to answer questions like *"show me all errors"* or *"any FTP-related warnings?"* from a 50K-row event log dataset.

## Technical Details

### Architecture

```
Browser (Chat UI)
      │
      ▼
tool_ui.py (Python)            ◄── langchain ReAct agent (gpt-4-turbo)
      │                            tools loaded from MCP server
      ▼
iris-mcp-server (Rust, :8080)
      │
      ▼
Sample.ToolSet (IRIS, IRISAPP namespace)
   ├── Sample.Tools          AddPerson, GetPeopleYoungerThan
   ├── PythonServer          multiply (multiplication_mcp.py)
   └── EventLogServer        query_iris_logs (event_log_mcp.py)  ◄── new
```

### What we built on top of the template

**1. New MCP tool `query_iris_logs`** ([src/Python/event_log_mcp.py](src/Python/event_log_mcp.py))
- Reads Ensemble event log — schema: `ID, Type, TimeLogged, SESSION, Job, SOURCE, Text, CLASS, METHOD, STACK`
- **Smart Python-side filter**: detects `error/warning/alert/trace/info` in the user's query and filters by `Type` column; if still too many rows, ranks by keyword overlap against `Text/SOURCE/CLASS/METHOD`
- File-based fallback source: reads IRIS `messages.log` + `^ERRORS` global via `iris.execute()`
- Filtered subset (capped at 80 rows) is passed to `Sample.Agent.Chat()` (the IRIS AI Hub agent) for plain-English summarisation, which goes through the IRIS OpenAI provider rather than direct API (the project key has zero RPM for direct `gpt-4o-mini` calls)
- CSV is cached in memory with mtime check — first call loads, subsequent calls are instant

**2. Agentic chatbot UI** ([tool_ui.py](tool_ui.py))
- Self-contained Python HTTP server on port 5000 (stdlib `http.server`, no Flask)
- **Chat tab**: langchain `create_agent` ReAct loop with all 4 MCP tools registered. Agent autonomously picks `query_iris_logs` for log questions, `AddPerson` for DB writes, `multiply` for math, etc. Every bot message shows the agent's tool-call trace inline so you can see *which* tool was picked and *what args* were passed.
- **Tool Tester tab**: dynamic forms auto-generated from each MCP tool's schema for direct invocation — the original tester, kept for debugging.

**3. Resolved IRIS quirks during development**
- `iris.cls()` doesn't honour runtime namespace switches in the embedded irispython we shipped with — replaced with `iris.execute()` + a temp global (`^IRISPyLogTmp`) for IPC between Python and ObjectScript.
- ObjectScript `<SYNTAX>` errors from underscore-prefixed local variables (`_ag` parses as concat operator with no left operand) — renamed to `tag/tsc/tprompt`.
- The deprecated `langgraph.prebuilt.create_react_agent` hangs on `ainvoke()` in v1.0+ — switched to `langchain.agents.create_agent`.

## Setup Instructions

### Prerequisites
- Docker + docker-compose
- Python 3.10+ on host (for the chat UI)
- IRIS AI Hub container image (download from EAP portal — see template instructions below)
- OpenAI API key with access to `gpt-4-turbo` (or `gpt-3.5-turbo`)

### One-time setup

```bash
git clone https://github.com/intersystems-ready-hackathon/ready-2026-team-08.git
cd ready-2026-team-08

# Add your OpenAI key
echo 'OPENAI_API_KEY="sk-..."' > .env

# Build & start IRIS
docker-compose up -d --build

# Install host-side Python deps for the chat UI
pip install langchain langchain-openai langchain-mcp-adapters langgraph
```

### Run

In **terminal 1** — start the iris-mcp-server bridge (inside the container):
```bash
docker-compose exec -it iris bash
iris-mcp-server -c /home/irisowner/dev/config.toml run
```

In **terminal 2** — start the chat UI on the host:
```bash
python tool_ui.py
```

Open **http://localhost:5000**. The Chat tab is the default. Try:
- *Show me all errors in the logs*
- *Were there any FTP-related warnings?*
- *Add a person named Alice aged 30*
- *Multiply 12 by 7*

### Optional: change the agent model

The agent defaults to `gpt-4-turbo`. Override with:
```bash
AGENT_MODEL=gpt-3.5-turbo python tool_ui.py
```

## Publicly accessible statement

We are happy for our project to be publicly visible after the event (we will remain repo admins).

---

# Template Instructions (kept for reference)

This repo started from the AI Hub template.

## Contents

- **./skills** — agent skills with info on using AI Hub for AI agents.
- **./src/Sample** — sample classes for tools, toolsets, agents and MCP servers (loaded via zpm at container build).
- **./src/Python** — stdio MCP servers in Python, used by IRIS Toolsets. We added `event_log_mcp.py`.
- **Datasets.md** — notes on Open Exchange datasets.

## Download AI Hub Container

1. Download an AI Hub container from the [Early Access Program Portal](https://evaluation.intersystems.com/Eval/early-access/AIHub) — pick the right arch (`arm64` for macOS).

2. Load the image:
    ```bash
    docker load -i /path/to/irishealth-community-2026.2.0AI.158.0-docker.tar.gz
    ```
    Confirm with `docker images`.

3. Match the image name in [Dockerfile](./Dockerfile) to the loaded image.

## Accessing IRIS

- Management Portal: <http://localhost:52773/csp/sys/UtilHome.csp> (login `SuperUser` / `SYS`)
- IRIS Terminal: `docker-compose exec -it iris iris session iris`
- Bash inside container: `docker-compose exec -it iris bash`

## Testing the sample IRIS agent (ObjectScript path)

```objectscript
set $NAMESPACE = "IRISAPP"
Set agent = ##class(Sample.Agent).%New()
Set sc = agent.%Init()
write:sc'=1 $SYSTEM.Status.GetErrorText(sc), !

Set session = agent.CreateSession()
Set request = "Add a person named Alice aged 30, and then get people younger than 35."
Set response = agent.Chat(session, request)
write response.content
```

## MCP server endpoints

- Service discovery: <http://localhost:52773/mcp/sample/v1/services>
- HTTP transport (`iris-mcp-server` running): <http://localhost:8080/mcp/sample>

A minimal Python MCP client example is in [test_mcp.py](./test_mcp.py).
