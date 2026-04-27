"""
MCP Tool Tester + Agentic Chat UI
Run:  python tool_ui.py
Open: http://localhost:5000
"""
import asyncio, base64, json, os, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_URL  = "http://localhost:8080/mcp/sample"
AUTH     = base64.b64encode(b"SuperUser:SYS").decode()
UI_PORT  = 5000
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4-turbo")

# ── Load .env (OPENAI_API_KEY) without any external dep ──────────────────────
def _load_env():
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)

_load_env()


# ── MCP helpers (mirror test_mcp.py) ─────────────────────────────────────────

def _make_client():
    return MultiServerMCPClient({
        "sample": {
            "transport": "http",
            "url": MCP_URL,
            "headers": {"Authorization": f"Basic {AUTH}"},
        }
    })

async def _list_tools():
    return await _make_client().get_tools()

async def _call_tool(name, args):
    tools = await _make_client().get_tools()
    tool  = next((t for t in tools if t.name == name), None)
    if tool is None:
        raise ValueError(f"Tool '{name}' not found")
    return await tool.ainvoke(args)


# ── Agentic chat (langgraph ReAct + MCP tools) ───────────────────────────────

SYSTEM_PROMPT = """You are an InterSystems IRIS assistant.

You have access to several tools exposed by the IRIS MCP server. Pick the
right tool based on the user's question:

  • mcp_sample_query_iris_logs    — for ANY question about IRIS event logs,
                                     errors, warnings, alerts, traces, or
                                     activity in the system.
  • mcp_sample_AddPerson          — add a person to the IRIS database.
  • mcp_sample_GetPeopleYoungerThan — query people by age.
  • mcp_sample_multiply           — multiply two integers.

Rules:
  - When the user asks about logs, errors, warnings, system events, or
    anything that sounds operational, ALWAYS call mcp_sample_query_iris_logs
    with their question as the `query` argument and `source="csv"`.
  - Do not invent data. If a tool returns nothing, say so plainly.
  - Be concise and friendly. Quote specific log entries when relevant.
"""


async def _agent_chat(user_message: str):
    """Run a single ReAct agent turn. Returns dict with answer + step trace."""
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"answer": "OPENAI_API_KEY is not set. Add it to .env.", "steps": []}

    print(f"  [agent] loading MCP tools…")
    tools = await _make_client().get_tools()
    print(f"  [agent] {len(tools)} tools loaded: {[t.name for t in tools]}")

    llm   = ChatOpenAI(model=AGENT_MODEL, api_key=api_key, temperature=0)
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    print(f"  [agent] invoking on: {user_message!r}")
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": user_message}]}),
            timeout=120,
        )
    except asyncio.TimeoutError:
        return {"answer": "Agent timed out after 120s. Try a simpler question or switch model.", "steps": []}
    print(f"  [agent] done — result keys: {list(result.keys())}")

    msgs  = result.get("messages", [])
    print(f"  [agent] {len(msgs)} messages in result")
    for i, m in enumerate(msgs):
        t = getattr(m, "type", None) or type(m).__name__
        c = getattr(m, "content", "")
        c_preview = (c[:80] if isinstance(c, str) else str(c)[:80])
        tc = getattr(m, "tool_calls", None) or []
        print(f"    msg[{i}] type={t} content={c_preview!r} tool_calls={len(tc)}")

    # ── Extract steps + final answer ──
    steps = []
    answer = ""

    def _content_to_text(c):
        """Handle both str content and list-of-content-blocks (newer models)."""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict):
                    parts.append(block.get("text") or block.get("content") or "")
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(c) if c else ""

    for m in msgs:
        t = getattr(m, "type", None) or ""
        content = _content_to_text(getattr(m, "content", "") or "")
        tool_calls = getattr(m, "tool_calls", None) or []

        if t == "ai":
            for tc in tool_calls:
                steps.append({
                    "kind": "tool_call",
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?"),
                    "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}),
                })
            if content:
                answer = content   # last AI text wins
        elif t == "tool":
            preview = content if len(content) < 800 else content[:800] + "…"
            steps.append({
                "kind": "tool_result",
                "name": getattr(m, "name", "?"),
                "content": preview,
            })

    if not answer:
        answer = "(agent produced no text answer — check server log)"

    print(f"  [agent] final answer ({len(answer)} chars): {answer[:120]!r}")
    print(f"  [agent] steps: {len(steps)}")

    return {"answer": answer, "steps": steps, "model": AGENT_MODEL}


# ── HTML page ────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IRIS MCP — Tools + Chat</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f1117; color: #e2e8f0; }

  /* ── Header / tabs ── */
  header { background: #1a1d27; border-bottom: 1px solid #2d3148; padding: 14px 28px;
           display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 10; }
  header h1 { font-size: 17px; color: #a78bfa; font-weight: 600; }
  header .sub { font-size: 12px; color: #6b7280; }
  .tabs { display: flex; gap: 4px; margin-left: 24px; }
  .tab { padding: 7px 16px; border-radius: 7px; font-size: 13px; font-weight: 500;
         background: transparent; color: #6b7280; border: 1px solid transparent;
         cursor: pointer; transition: all .12s; font-family: inherit; }
  .tab.active { background: #21253a; color: #a78bfa; border-color: #2d3148; }
  .tab:hover:not(.active) { color: #a0aec0; }
  .badge { margin-left: auto; background: #1e3a5f; color: #60a5fa; border: 1px solid #1d4ed8;
           padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }

  /* ── View switcher ── */
  .view { display: none; }
  .view.active { display: flex; flex-direction: column; }

  /* ── Tools view ── */
  #view-tools { max-width: 720px; margin: 28px auto; padding: 0 16px; gap: 18px; }
  .card { background: #1a1d27; border: 1px solid #2d3148; border-radius: 10px; overflow: hidden; }
  .card-header { padding: 13px 18px; background: #21253a; border-bottom: 1px solid #2d3148;
                  display: flex; align-items: center; gap: 10px; }
  .tool-icon { font-size: 20px; }
  .tool-name { font-size: 15px; font-weight: 600; color: #c4b5fd; }
  .tool-desc { font-size: 12px; color: #6b7280; margin-top: 2px; }
  .card-body { padding: 16px; display: flex; flex-direction: column; gap: 12px; }

  label { display: block; font-size: 12px; color: #8b95b0; margin-bottom: 5px; font-weight: 500; }
  input, textarea { width: 100%; background: #0f1117; border: 1px solid #2d3148; color: #e2e8f0;
                    padding: 9px 12px; border-radius: 7px; font-size: 14px; font-family: inherit; }
  input:focus, textarea:focus { outline: none; border-color: #7c3aed; }
  textarea { resize: vertical; min-height: 60px; }

  button.run-btn { background: #7c3aed; color: white; border: none; border-radius: 7px;
                    padding: 9px 18px; font-size: 13px; font-weight: 600; cursor: pointer;
                    font-family: inherit; align-self: flex-start; transition: background .12s; }
  button.run-btn:hover:not(:disabled) { background: #6d28d9; }
  button:disabled { opacity: .5; cursor: not-allowed; }

  .result { background: #0f1117; border: 1px solid #2d3148; border-radius: 7px; padding: 12px;
             font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 12.5px;
             white-space: pre-wrap; word-break: break-word; color: #a0aec0; display: none; }
  .result.ok  { border-color: #065f46; color: #6ee7b7; display: block; }
  .result.err { border-color: #7f1d1d; color: #fca5a5; display: block; }

  /* ── Chat view ── */
  #view-chat { flex: 1; min-height: 0; display: none; }
  #view-chat.active { display: flex; }
  .chat-wrap { max-width: 760px; width: 100%; margin: 0 auto; padding: 22px 16px 0;
                flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .chat-log { flex: 1; overflow-y: auto; padding: 8px 4px 16px; display: flex;
              flex-direction: column; gap: 14px; }
  .msg { display: flex; gap: 10px; align-items: flex-start; max-width: 82%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .msg .avatar { width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
                  display: flex; align-items: center; justify-content: center;
                  font-size: 14px; font-weight: 600; }
  .msg.user .avatar { background: #1e3a5f; color: #60a5fa; }
  .msg.bot  .avatar { background: #2d1f7a; color: #c4b5fd; }
  .bubble { background: #1a1d27; border: 1px solid #2d3148; border-radius: 12px;
             padding: 10px 14px; font-size: 14px; line-height: 1.5; color: #e2e8f0;
             white-space: pre-wrap; word-break: break-word; }
  .msg.user .bubble { background: #1e3a5f; border-color: #1d4ed8; color: #dbeafe; }

  .steps { margin-top: 8px; font-size: 11px; color: #6b7280; border-left: 2px solid #2d3148;
            padding-left: 10px; display: flex; flex-direction: column; gap: 3px; }
  .step { display: flex; gap: 6px; align-items: center; }
  .step .pill { background: #2d1f7a; color: #c4b5fd; padding: 1px 7px; border-radius: 10px;
                font-weight: 600; font-size: 10px; }
  .step .pill.tool-result { background: #052e16; color: #6ee7b7; }
  .step code { background: #0f1117; color: #fbbf24; padding: 1px 5px; border-radius: 3px;
                font-size: 10.5px; }

  .typing { font-style: italic; color: #6b7280; font-size: 13px; }
  .typing::after { content: '▍'; animation: blink 1s infinite; }
  @keyframes blink { 0%,49%{opacity:1} 50%,100%{opacity:0} }

  /* ── Chat input ── */
  .chat-input { border-top: 1px solid #2d3148; background: #13151f; padding: 14px 16px;
                 display: flex; gap: 10px; align-items: flex-end; }
  .chat-input textarea { flex: 1; min-height: 44px; max-height: 160px; resize: none;
                          line-height: 1.5; }
  .chat-input button { background: #7c3aed; color: white; border: none; border-radius: 7px;
                        padding: 11px 20px; font-size: 14px; font-weight: 600; cursor: pointer;
                        font-family: inherit; transition: background .12s; }
  .chat-input button:hover:not(:disabled) { background: #6d28d9; }

  .empty-state { text-align: center; color: #4b5563; padding: 40px 20px; font-size: 14px; }
  .empty-state .icon { font-size: 36px; margin-bottom: 12px; }
  .examples { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 14px; }
  .example { background: #21253a; border: 1px solid #2d3148; color: #a0aec0;
              padding: 6px 12px; border-radius: 14px; font-size: 12px; cursor: pointer;
              transition: all .12s; }
  .example:hover { background: #2d3148; color: #e2e8f0; border-color: #7c3aed; }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
</style>
</head>
<body>

<header>
  <div>
    <h1>IRIS MCP</h1>
    <div class="sub">Sample ToolSet via Agentic Chat</div>
  </div>
  <div class="tabs">
    <button class="tab active" data-view="chat" onclick="switchView('chat')">💬 Chat</button>
    <button class="tab" data-view="tools" onclick="switchView('tools')">🔧 Tool Tester</button>
  </div>
  <div class="badge" id="badge">loading…</div>
</header>

<!-- ═══════════ Chat view ═══════════ -->
<div id="view-chat" class="view active">
  <div class="chat-wrap">
    <div class="chat-log" id="chatLog">
      <div class="empty-state">
        <div class="icon">💬</div>
        Ask me anything about IRIS event logs, errors, or operations.
        <div class="examples">
          <div class="example" onclick="runExample(this)">Show me all errors in the logs</div>
          <div class="example" onclick="runExample(this)">Were there any FTP-related warnings?</div>
          <div class="example" onclick="runExample(this)">What kinds of events occurred today?</div>
          <div class="example" onclick="runExample(this)">Add a person named Alice age 30</div>
          <div class="example" onclick="runExample(this)">Multiply 12 by 7</div>
        </div>
      </div>
    </div>
    <div class="chat-input">
      <textarea id="chatInput" placeholder="Ask about IRIS logs, errors, or operations…"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage();}"></textarea>
      <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
  </div>
</div>

<!-- ═══════════ Tools view ═══════════ -->
<div id="view-tools" class="view">
  <div id="toolsHost"><p style="color:#6b7280;text-align:center;padding:40px;">Loading tools…</p></div>
</div>

<script>
// ─── Shared state ────────────────────────────────────────────────────────────
const ICONS = { AddPerson:'👤', GetPeople:'🔍', multiply:'✖', query_iris_logs:'📜' };
const schemas = {};

function icon(name) {
  for (const [k,v] of Object.entries(ICONS)) if (name.includes(k)) return v;
  return '🔧';
}

function switchView(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === name));
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name));
}

// ─── Tools loading (used by both Tool Tester and the agent badge) ────────────
async function loadTools() {
  try {
    const r = await fetch('/api/tools');
    if (!r.ok) throw new Error(await r.text());
    const tools = await r.json();
    document.getElementById('badge').textContent = tools.length + ' tool' + (tools.length !== 1 ? 's' : '');
    renderTools(tools);
  } catch(e) {
    document.getElementById('toolsHost').innerHTML =
      '<div style="color:#fca5a5;padding:30px;text-align:center;border:1px solid #7f1d1d;border-radius:8px;margin:20px;">' +
      '⚠ Could not reach MCP server: ' + e.message + '</div>';
    document.getElementById('badge').textContent = 'offline';
  }
}

function renderTools(tools) {
  const host = document.getElementById('toolsHost');
  host.innerHTML = '';
  const TEXTAREA_KEYS = ['query','question','message','prompt','text','input'];

  tools.forEach(t => {
    const props = t.args && t.args.properties ? Object.entries(t.args.properties) : [];
    const req   = (t.args && t.args.required) || [];
    schemas[t.name] = { props, req };

    const fields = props.map(([k, def]) => {
      const type = def.type || 'string';
      const isTextarea = type === 'string' && TEXTAREA_KEYS.some(kw => k.toLowerCase().includes(kw));
      const isNumber   = type === 'integer' || type === 'number';
      const input = isTextarea
        ? `<textarea id="${t.name}__${k}" rows="2" placeholder="${def.description || k}"></textarea>`
        : `<input id="${t.name}__${k}" type="${isNumber?'number':'text'}"
                  placeholder="${def.description || k}" ${type==='integer'?'step=1':''}/>`;
      return `<div><label>${k}${req.includes(k)?' <span style="color:#f87171">*</span>':''}
              <span style="font-weight:400;color:#4b5563;">(${type})</span></label>${input}</div>`;
    }).join('');

    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.tool = t.name;
    card.innerHTML = `
      <div class="card-header">
        <span class="tool-icon">${icon(t.name)}</span>
        <div>
          <div class="tool-name">${t.name}</div>
          <div class="tool-desc">${t.description || ''}</div>
        </div>
      </div>
      <div class="card-body">
        ${fields}
        <button class="run-btn">▶ Run</button>
        <pre class="result" id="res__${t.name}"></pre>
      </div>`;
    card.querySelector('.run-btn').addEventListener('click', () => invokeTool(t.name));
    host.appendChild(card);
  });
}

async function invokeTool(toolName) {
  const { props } = schemas[toolName] || { props: [] };
  const args = {};
  for (const [k, def] of props) {
    const el = document.getElementById(toolName + '__' + k);
    if (!el) continue;
    let v = el.value.trim();
    const type = def.type || 'string';
    if (type === 'integer' || type === 'number') v = Number(v);
    if (v !== '' || type !== 'string') args[k] = v;
  }
  const card = document.querySelector(`[data-tool="${toolName}"]`);
  const btn  = card.querySelector('.run-btn');
  const res  = document.getElementById('res__' + toolName);
  btn.disabled = true; btn.textContent = 'Running…';
  res.className = 'result'; res.textContent = '';

  try {
    const r = await fetch('/api/invoke', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ tool: toolName, args })
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    res.className = 'result ok';
    res.textContent = typeof data.result === 'string' ? data.result : JSON.stringify(data.result, null, 2);
  } catch(e) {
    res.className = 'result err';
    res.textContent = e.message;
  } finally {
    btn.disabled = false; btn.textContent = '▶ Run';
  }
}

// ─── Chat ────────────────────────────────────────────────────────────────────
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function clearEmptyState() {
  const empty = document.querySelector('#chatLog .empty-state');
  if (empty) empty.remove();
}

function appendUser(text) {
  clearEmptyState();
  const log = document.getElementById('chatLog');
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `<div class="avatar">U</div><div class="bubble">${escHtml(text)}</div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function appendBotTyping() {
  const log = document.getElementById('chatLog');
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.id = 'typing-msg';
  div.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><span class="typing">thinking</span></div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function replaceBotTyping(answer, steps) {
  const div = document.getElementById('typing-msg');
  if (!div) return;
  div.removeAttribute('id');

  const stepsHtml = (steps && steps.length)
    ? '<div class="steps">' + steps.map(s => {
        if (s.kind === 'tool_call') {
          const argStr = s.args ? Object.entries(s.args).map(([k,v]) =>
            `${k}=<code>${escHtml(typeof v === 'string' ? (v.length>40?v.slice(0,40)+'…':v) : JSON.stringify(v))}</code>`
          ).join(' ') : '';
          return `<div class="step"><span class="pill">→ tool</span><code>${escHtml(s.name)}</code> ${argStr}</div>`;
        }
        if (s.kind === 'tool_result') {
          const preview = (s.content || '').replace(/\\s+/g, ' ').slice(0, 110);
          return `<div class="step"><span class="pill tool-result">← result</span><code>${escHtml(s.name)}</code> <span style="color:#6b7280;">${escHtml(preview)}…</span></div>`;
        }
        return '';
      }).join('') + '</div>'
    : '';

  div.querySelector('.bubble').innerHTML = escHtml(answer) + stepsHtml;
  document.getElementById('chatLog').scrollTop = document.getElementById('chatLog').scrollHeight;
}

function appendBotError(msg) {
  const div = document.getElementById('typing-msg');
  if (!div) return;
  div.removeAttribute('id');
  div.querySelector('.bubble').innerHTML = `<span style="color:#fca5a5;">⚠ ${escHtml(msg)}</span>`;
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const btn   = document.getElementById('sendBtn');
  const text  = input.value.trim();
  if (!text) return;

  input.value = '';
  btn.disabled = true;
  appendUser(text);
  appendBotTyping();

  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ message: text })
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    replaceBotTyping(data.answer || '(no answer)', data.steps || []);
  } catch(e) {
    appendBotError(e.message);
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

function runExample(el) {
  document.getElementById('chatInput').value = el.textContent;
  sendMessage();
}

loadTools();
document.getElementById('chatInput').focus();
</script>
</body>
</html>"""

# ── HTTP handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {fmt % args}")

    def _json(self, code, data):
        try:
            body = json.dumps(data, default=str).encode()
        except Exception as e:
            body = json.dumps({"error": f"Serialize failed: {e}"}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        try:
            self.wfile.flush()
        except Exception:
            pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/tools":
            try:
                raw = asyncio.run(_list_tools())
                tools = []
                for t in raw:
                    props = dict(getattr(t, "args", {}) or {})
                    req   = list(props.keys())
                    tools.append({
                        "name": t.name,
                        "description": t.description or "",
                        "args": {"properties": props, "required": req},
                    })
                self._json(200, tools)
            except Exception as e:
                self._json(500, {"error": str(e)})
        else:
            self.send_error(404)

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/api/invoke":
            try:
                result = asyncio.run(_call_tool(payload.get("tool", ""), payload.get("args", {})))
                self._json(200, {"result": result})
            except Exception as e:
                self._json(200, {"error": str(e)})

        elif self.path == "/api/chat":
            try:
                data = asyncio.run(_agent_chat(payload.get("message", "")))
                self._json(200, data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._json(200, {"error": f"{type(e).__name__}: {e}"})

        else:
            self.send_error(404)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", UI_PORT), Handler)
    print(f"\n  IRIS MCP UI       →  http://localhost:{UI_PORT}")
    print(f"  MCP server        →  {MCP_URL}")
    print(f"  Agent model       →  {AGENT_MODEL}")
    print(f"  OPENAI_API_KEY    →  {'set' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}\n")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        sys.exit(0)
