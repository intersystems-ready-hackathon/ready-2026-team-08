import logging
import sys
import os
import re
import csv
import iris
from mcp.server.fastmcp import FastMCP

# stdout MUST contain only MCP protocol JSON — all other output goes to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger(__name__)

mcp = FastMCP("event_log", log_level="ERROR")

# ════════════════════════════════════════════════════════════════════
# CSV reading (smart filtering for the eventlog (3).csv file)
# ════════════════════════════════════════════════════════════════════

CSV_PATH = os.environ.get(
    "EVENTLOG_CSV",
    "/home/irisowner/dev/src/Sample/eventlog (3).csv",
)

# CSV columns (after the --ROW header line is skipped):
# 0:ID  1:Type  2:TimeLogged  3:SESSION  4:Job  5:SOURCE  6:Text  7:CLASS  8:METHOD  9:STACK
COLS = ("ID", "Type", "TimeLogged", "SESSION", "Job", "SOURCE", "Text", "CLASS", "METHOD", "STACK")

_csv_cache = {"mtime": 0, "rows": []}


def _load_csv() -> list:
    """Load (and cache) the CSV file. Reload only if mtime changes."""
    if not os.path.exists(CSV_PATH):
        return []

    mtime = os.path.getmtime(CSV_PATH)
    if _csv_cache["rows"] and mtime == _csv_cache["mtime"]:
        return _csv_cache["rows"]

    rows = []
    with open(CSV_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        first = next(reader, None)
        # If the first row is the --ROW(...) schema header, skip it.
        # Otherwise, treat it as data.
        if first and not (first[0].startswith("--ROW") or first[0].startswith("ID")):
            rows.append(first)
        for row in reader:
            if len(row) >= 9:
                rows.append(row)

    _csv_cache["mtime"] = mtime
    _csv_cache["rows"] = rows
    log.info("Loaded %d rows from %s", len(rows), CSV_PATH)
    return rows


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "what", "when", "where", "why", "how", "who", "which",
    "show", "tell", "give", "list", "find", "any", "some", "all",
    "me", "us", "you", "of", "in", "on", "at", "for", "to", "from",
    "and", "or", "but", "with", "about", "into", "have", "has", "had",
    "do", "does", "did", "log", "logs", "iris",
}


def _filter_rows(rows: list, query: str, max_rows: int) -> list:
    """Smart filter: detect Type from query keywords, then keyword search Text/SOURCE/CLASS/METHOD."""
    q = query.lower()

    # 1) Type filter
    type_filter = None
    if re.search(r"\berror", q):
        type_filter = "Error"
    elif re.search(r"\bwarn", q):
        type_filter = "Warning"
    elif re.search(r"\balert", q):
        type_filter = "Alert"
    elif re.search(r"\btrace", q):
        type_filter = "Trace"
    elif re.search(r"\binfo", q):
        type_filter = "Info"

    pool = rows
    if type_filter:
        pool = [r for r in rows if len(r) > 1 and r[1] == type_filter]

    # 2) If still too many rows, narrow by keyword match
    if len(pool) > max_rows:
        keywords = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_.]{2,}", q)
                    if w.lower() not in _STOPWORDS]
        if keywords:
            kw_lower = [k.lower() for k in keywords]
            scored = []
            for r in pool:
                blob = " ".join(r[i] for i in (5, 6, 7, 8) if i < len(r)).lower()
                hits = sum(1 for k in kw_lower if k in blob)
                if hits:
                    scored.append((hits, r))
            if scored:
                scored.sort(key=lambda x: -x[0])
                pool = [r for _, r in scored]

    # 3) Cap to max_rows. The CSV is sorted newest-first by ID, so this gives recent.
    return pool[:max_rows]


def _format_rows(rows: list) -> str:
    """Format rows for AI consumption — compact, one row per line, key fields only."""
    if not rows:
        return "(no matching rows)"

    out = []
    for r in rows:
        # Pad short rows
        r = list(r) + [""] * (len(COLS) - len(r))
        ts     = r[2] or ""
        typ    = r[1] or ""
        src    = r[5] or ""
        text   = (r[6] or "").replace("\n", " ").replace("\t", " ")
        cls    = r[7] or ""
        method = r[8] or ""
        line = f"[{ts}] {typ:<7} src={src} cls={cls}.{method}  -- {text}"
        # Truncate insanely long lines
        if len(line) > 600:
            line = line[:600] + "…"
        out.append(line)
    return "\n".join(out)


def _csv_summary(rows: list) -> str:
    """Quick stats on the full CSV for context."""
    counts = {}
    for r in rows:
        if len(r) > 1:
            counts[r[1]] = counts.get(r[1], 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return f"Total rows={len(rows)}  ({', '.join(parts)})"


# ════════════════════════════════════════════════════════════════════
# File-based source (messages.log + ^ERRORS) — kept for compatibility
# ════════════════════════════════════════════════════════════════════

def _mgr_dir() -> str:
    for c in ("/usr/irissys/mgr", "/usr/iris/mgr"):
        if os.path.isdir(c):
            return c
    return "/usr/irissys/mgr"


def _read_messages_log(max_lines: int) -> str:
    mgr = _mgr_dir()
    for name in ("messages.log", "cconsole.log"):
        path = os.path.join(mgr, name)
        if os.path.exists(path):
            try:
                with open(path, "r", errors="replace") as f:
                    lines = f.readlines()
                return "".join(lines[-max_lines:])
            except Exception as e:
                return f"Error reading {path}: {e}"
    return f"Log file not found in {mgr}"


# ════════════════════════════════════════════════════════════════════
# AI analysis via Sample.Agent (run inline in iris.execute() block)
# Uses non-underscore variable names to avoid <SYNTAX> from `_var` parsing.
# ════════════════════════════════════════════════════════════════════

def _analyze_with_iris_ai(query: str, logs: str) -> str:
    truncated = logs[:8000]
    g = iris.gref("^IRISPyLogTmp")
    try:
        g["q"] = query
        g["l"] = truncated

        iris.execute(
            'New $namespace Set $namespace = "IRISAPP" '
            'Set tag = ##class(Sample.Agent).%New() '
            'Set tsc = tag.%CreateProvider() '
            'Set tprompt = "You are an IRIS log analyst. Do NOT call any tools. '
            'Answer using only the data below. Be concise. List specific entries when asked." '
            '_$CHAR(10,10)_"Question: "_^IRISPyLogTmp("q") '
            '_$CHAR(10,10)_"Log Data:"_$CHAR(10)_^IRISPyLogTmp("l") '
            'Set ^IRISPyLogTmp("r") = tag.Chat(tprompt)'
        )

        result = g["r"]
        iris.execute("Kill ^IRISPyLogTmp")
        return str(result) if result else "(empty AI response)"

    except Exception as e:
        log.error("AI call failed: %s", e)
        try:
            iris.execute("Kill ^IRISPyLogTmp")
        except Exception:
            pass
        return f"[AI unavailable: {e}]\n\nRaw filtered data:\n\n{logs}"


# ════════════════════════════════════════════════════════════════════
# Source dispatch
# ════════════════════════════════════════════════════════════════════

def _collect_logs(query: str, source: str, max_rows: int) -> str:
    if source == "file":
        return f"=== messages.log (last {max_rows} lines) ===\n{_read_messages_log(max_rows)}"

    # Default: csv
    rows = _load_csv()
    if not rows:
        return f"CSV not found at {CSV_PATH}"
    summary  = _csv_summary(rows)
    filtered = _filter_rows(rows, query, max_rows)
    return f"=== eventlog CSV ===\n{summary}\n\nFiltered rows ({len(filtered)}):\n{_format_rows(filtered)}"


# ════════════════════════════════════════════════════════════════════
# MCP Tool
# ════════════════════════════════════════════════════════════════════

@mcp.tool()
async def query_iris_logs(query: str, source: str = "csv", max_rows: int = 80) -> str:
    """Ask a natural-language question about IRIS event logs.

    Reads logs from the chosen source, filters them based on the query,
    and uses an IRIS AI agent to answer in plain English.

    Args:
        query: What you want to know — e.g. 'Show me all errors',
               'Which sources had warnings?', 'Any FTP-related issues?'
        source: 'csv' (default) reads the eventlog (3).csv file with smart
                filtering by Type/keyword. 'file' reads IRIS messages.log.
        max_rows: Cap on rows passed to AI after filtering (default 80).
    """
    log.info("query_iris_logs source=%s query=%r", source, query)
    logs = _collect_logs(query, source.lower().strip(), max_rows)
    return _analyze_with_iris_ai(query, logs)


if __name__ == "__main__":
    mcp.run(transport="stdio")
