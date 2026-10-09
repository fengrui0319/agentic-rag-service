from __future__ import annotations
import argparse
import difflib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from mcp_client import MCPClient, to_model_tool, parse_mcp_tool_name
from skills import build_skill_catalog, read_skill
from subagent import run_subagent
from rag import retrieve_knowledge
import re

load_dotenv()

#1.config
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com/anthropic",
)
MAX_TURNS = 50
SESSION_DIR = Path.home() / ".agentic-rag-service" / "projects"
DEFAULT_PERMISSION = os.environ.get("AGENT_PERMISSION", "ask")  # ask|allow|deny
USE_STREAM = True

# DeepSeek Flash prices, CNY per 1M tokens.
# DeepSeek uses automatic context caching.
# Peak/off-peak pricing may change; verify against official pricing docs.
DEEPSEEK_PRICE_TIER = os.environ.get("DEEPSEEK_PRICE_TIER", "peak").lower()

DEEPSEEK_PRICES = {
    "peak": {
        "input_miss": 2.0,
        "cache_hit": 0.04,
        "output": 8.0,
    },
    "offpeak": {
        "input_miss": 1.0,
        "cache_hit": 0.02,
        "output": 4.0,
    },
}

if DEEPSEEK_PRICE_TIER not in DEEPSEEK_PRICES:
    raise ValueError("DEEPSEEK_PRICE_TIER must be 'peak' or 'offpeak'")

PRICES = DEEPSEEK_PRICES[DEEPSEEK_PRICE_TIER]

CONTEXT_WINDOW = 1_000_000
COMPACT_TRIGGER = 0.60
KEEP_RECENT = 6

console = Console()

#2.tools
def tool_read(file_path: str, **_):
    p = Path(file_path).expanduser()
    if not p.exists():
        return f"ERROR: no such file: {file_path}"
    if not p.is_file():
        return f"ERROR: not a file: {file_path}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:50_000]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def tool_write(file_path: str, content: str, **_):
    p = Path(file_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {file_path}"

def tool_edit(file_path: str, old_string: str, new_string: str, **_):
    p = Path(file_path).expanduser()
    if not p.exists():
        return f"ERROR: no such file: {file_path}"
    body = p.read_text(encoding="utf-8")
    if old_string not in body:
        return f"ERROR: old_string not found in {file_path}"
    if body.count(old_string) > 1:
        return (f"ERROR: old_string appears {body.count(old_string)} times — "
                f"add surrounding context to make it unique")
    new_body = body.replace(old_string, new_string)
    p.write_text(new_body, encoding="utf-8")
    return _diff_summary(file_path, body, new_body)

def tool_bash(command: str, **_):
    try:
        r = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=120)
        out = (r.stdout + r.stderr)[:20_000]
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 120s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def tool_glob(pattern: str, path: str = ".", **_):
    try:
        matches = sorted(Path(path).expanduser().glob(pattern))

        return "\n".join(str(p) for p in matches[:200]) or "(no matches)"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def tool_grep(pattern: str, path: str = ".", **_):
    """Use ripgrep if available; fall back to grep."""
    for cmd in (["rg", "-n", pattern, path], ["grep", "-rn", pattern, path]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return r.stdout[:20_000] or "(no matches)"
        except FileNotFoundError:
            continue
    return "ERROR: neither rg nor grep available"

# Keep TodoWrite state on the Session object across in-process compaction.
def tool_todo_write(todos: list, session=None, **_):
    if session is not None:
        session.todos = todos
    rendered = "\n".join(
        f"  [{t.get('status','pending')[0].upper()}] {t.get('content','')}"
        for t in todos)
    return f"todos updated ({len(todos)} items):\n{rendered}"

TOOLS = [
    {"name": "Read",
     "description": "Read a file's full contents. Returns up to 50KB of UTF-8 text.",
     "input_schema": {"type": "object",
                      "properties": {"file_path": {"type": "string"}},
                      "required": ["file_path"]}},
    {"name": "Write",
     "description": "Write a file. Creates parent directories. Overwrites if exists.",
     "input_schema": {"type": "object",
                      "properties": {"file_path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["file_path", "content"]}},
    {"name": "Edit",
     "description": ("Replace exact string old_string with new_string in a file. "
                     "old_string must appear EXACTLY ONCE in the file — include "
                     "surrounding lines for context if needed."),
     "input_schema": {"type": "object",
                      "properties": {"file_path": {"type": "string"},
                                     "old_string": {"type": "string"},
                                     "new_string": {"type": "string"}},
                      "required": ["file_path", "old_string", "new_string"]}},
    {"name": "Bash",
     "description": ("Execute a shell command. Output truncated to 20KB. "
                     "Timeout 120s. The user is asked for permission first."),
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "Glob",
     "description": "Find files matching a glob pattern (e.g. '**/*.py').",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"},
                                     "path": {"type": "string",
                                              "description": "search root, default '.'"}},
                      "required": ["pattern"]}},
    {"name": "Grep",
     "description": "Search file contents with a regex (uses ripgrep if available).",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"},
                                     "path": {"type": "string",
                                              "description": "search root, default '.'"}},
                      "required": ["pattern"]}},
    {"name": "TodoWrite",
     "description": ("Track a multi-step plan. Call this when starting any task with "
                     "more than 2 steps. Each todo: {content: str, status: pending|in_progress|completed}. "
                     "Update by passing the WHOLE list each time."),
     "input_schema": {"type": "object",
                      "properties": {"todos": {"type": "array",
                                               "items": {"type": "object"}}},
                      "required": ["todos"]}},
    {"name": "read_skill",
     "description": ("Load the full instructions for an installed skill by name. "
                     "Use this when the skill catalog says a skill is relevant."),
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string",
                                              "description": "Skill name from the installed skill catalog"}},
                      "required": ["name"]}},
    {"name": "Task",
     "description": ("Delegate a self-contained research or code-inspection task "
                     "to a fresh read-only subagent. Use this when the task would "
                     "require reading many files or gathering lots of temporary "
                     "context that the parent does not need to keep. "
                     "The subagent returns only its final answer."),
     "input_schema": {"type": "object",
                      "properties": {"description": {"type": "string",
                                                    "description": ("A complete, self-contained description of "
                                                                    "what the subagent should investigate.")}},
                      "required": ["description"]}},
    {"name": "retrieve_knowledge",
     "description": ("Search the local project knowledge base for relevant "
                     "documentation. Return matching passages with source information. "
                     "Use this for questions about the project's documented features."),
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string",
                                               "description": "Search query"}},
                      "required": ["query"]}},
]

DISPATCH = {
    "Read":      tool_read,
    "Write":     tool_write,
    "Edit":      tool_edit,
    "Bash":      tool_bash,
    "Glob":      tool_glob,
    "Grep":      tool_grep,
    "TodoWrite": tool_todo_write,
    "read_skill": read_skill,
    "retrieve_knowledge": retrieve_knowledge,
}

WRITE_TOOLS = {"Write", "Edit", "Bash"}

SUBAGENT_TOOL_NAMES = {"Read", "Glob", "Grep", "read_skill"}

def build_subagent_tools() -> list[dict]:
    return [
        tool
        for tool in TOOLS
        if tool["name"] in SUBAGENT_TOOL_NAMES
    ]

def build_subagent_dispatch() -> dict:
    return {
        name: DISPATCH[name]
        for name in SUBAGENT_TOOL_NAMES
    }

# MCP tools
MCP_CLIENTS: dict[str, MCPClient] = {}
MCP_TOOLS: list[dict] = []

def init_mcp_clients() -> None:
    """Start MCP servers and discover their tools."""
    configs = {
        "calc": [
            sys.executable,
            str(Path("mcp_servers/calculator_server.py").resolve()),
        ],
        "weather": [
            sys.executable,
            str(Path("mcp_servers/weather_server.py").resolve()),
        ],
    }

    for server_id, command in configs.items():
        client = MCPClient(server_id, command)
        client.initialize()

        MCP_CLIENTS[server_id] = client

        for tool in client.tools:
            MCP_TOOLS.append(
                to_model_tool(server_id, tool)
            )

def close_mcp_clients() -> None:
    """Close all running MCP server processes."""
    for client in MCP_CLIENTS.values():
        try:
            client.close()
        except Exception:
            pass

    MCP_CLIENTS.clear()
    MCP_TOOLS.clear()

#3.rendering
def _tool_label(name: str, args: dict) -> str:
    if name == "Bash":      return f"Bash$ {args.get('command','')[:80]}"
    if name == "Read":      return f"Read({args.get('file_path','')})"
    if name == "Write":     return f"Write({args.get('file_path','')})"
    if name == "Edit":      return f"Edit({args.get('file_path','')})"
    if name == "Glob":      return f"Glob({args.get('pattern','')})"
    if name == "Grep":      return f"Grep({args.get('pattern','')})"
    if name == "TodoWrite": return f"TodoWrite ({len(args.get('todos',[]))} items)"
    if name == "Task":
        desc = args.get("description", "")
        return f"Task({desc[:100]})"
    return f"{name}({args})"

def render_tool_call(name: str, args: dict):
    label = _tool_label(name, args)
    console.print(f"[bold green]●[/bold green] {label}")

def render_tool_result(name: str, result: str):
    if result.startswith("ERROR"):
        for line in result.splitlines()[:6]:
            console.print(f"  [red]⎿  {line}[/red]")
        return
    if name == "Edit":
        for line in result.splitlines()[:30]:
            if line.startswith("+++") or line.startswith("---"):
                console.print(f"  [dim]⎿  {line}[/dim]")
            elif line.startswith("+"):
                console.print(f"  ⎿  [green]{line}[/green]")
            elif line.startswith("-"):
                console.print(f"  ⎿  [red]{line}[/red]")
            else:
                console.print(f"  ⎿  {line}")
        return
    lines = result.splitlines()
    show = lines[:8]
    for line in show:
        console.print(f"  ⎿  {line}")
    if len(lines) > len(show):
        console.print(f"  ⎿  [dim]... ({len(lines) - len(show)} more lines)[/dim]")

def _diff_summary(path: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                fromfile=path, tofile=path, lineterm="", n=2)
    return "\n".join(list(diff)[:40])

def render_text(text: str):
    if text.strip():
        console.print(Markdown(text))
#4.permissions
def ask_permission(name: str, args: dict) -> bool:
    if DEFAULT_PERMISSION == "allow":
        return True
    if DEFAULT_PERMISSION == "deny":
        return False
    label = _tool_label(name, args)
    console.print(f"\n[yellow]allow {label}? [Y/n][/yellow] ", end="")
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("", "y", "yes")

#5.AGENT.md loader
def load_agent_md(start: Path) -> str:
    """Walk up from cwd, collect AGENT.md files (root-first, deepest-last)."""
    parts = []
    p = start.resolve()
    chain = list(reversed([p] + list(p.parents)))
    for d in chain:
        candidate = d / "AGENT.md"
        if candidate.is_file():
            parts.append(f"# from {candidate}\n{candidate.read_text(encoding='utf-8', errors='replace')}")
    user = Path.home() / ".agentic-rag-service" / "AGENT.md"
    if user.is_file():
        parts.insert(0, f"# user-scope ({user})\n{user.read_text(encoding='utf-8', errors='replace')}")
    return "\n\n".join(parts)[:25_000]

#6.cost meter
class Meter:
    """Track DeepSeek token usage and estimate API cost."""

    def __init__(self):
        self.input_miss = 0
        self.cache_hit = 0
        self.output = 0
        self.cny = 0.0
        self.turns = 0

    def add(self, usage):
        self.turns += 1

        input_miss = getattr(usage, "input_tokens", 0) or 0
        output = getattr(usage, "output_tokens", 0) or 0
        cache_hit = getattr(usage, "cache_read_input_tokens", 0) or 0

        self.input_miss += input_miss
        self.cache_hit += cache_hit
        self.output += output

        spent = (
            input_miss * PRICES["input_miss"]
            + cache_hit * PRICES["cache_hit"]
            + output * PRICES["output"]
        ) / 1_000_000

        self.cny += spent
        return spent

    def status(self) -> str:
        return (
            f"¥{self.cny:.6f}  ·  "
            f"input_miss {self.input_miss}  "
            f"cache_hit {self.cache_hit}  "
            f"out {self.output}  "
            f"tier {DEEPSEEK_PRICE_TIER}"
        )

#7.sessions
class Session:
    """Append-only JSONL session log with message replay support."""

    def __init__(self, session_id: str | None = None, cwd: Path | None = None):
        self.id = (
            uuid.uuid4().hex[:12]
            if session_id is None
            else session_id
        )

        if (
                not isinstance(self.id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}", self.id) is None
        ):
            raise ValueError(
                "Invalid session_id: use 8-64 letters, digits, '_' or '-', "
                "starting with a letter or digit."
            )

        self.cwd = cwd or Path.cwd()
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", str(self.cwd.resolve())).strip("-") or "_root"
        self.dir = SESSION_DIR / slug
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.id}.jsonl"
        self.messages: list[dict] = []
        self.todos: list[dict] = []
        if self.path.exists():
            self._replay()
        else:
            self._write({"type": "meta", "session_id": self.id,
                         "started": datetime.now(timezone.utc).isoformat(),
                         "cwd": str(self.cwd), "model": MODEL})

    def append_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._write({"type": "user", "content": text,
                     "ts": datetime.now(timezone.utc).isoformat()})

    def append_assistant(self, content_blocks):
        ser = [b.model_dump() if hasattr(b, "model_dump") else b
               for b in content_blocks]
        self.messages.append({"role": "assistant", "content": ser})
        self._write({"type": "assistant", "content": ser,
                     "ts": datetime.now(timezone.utc).isoformat()})

    def append_tool_results(self, results: list[dict]):
        self.messages.append({"role": "user", "content": results})
        self._write({"type": "tool_results", "content": results,
                     "ts": datetime.now(timezone.utc).isoformat()})

    def truncate_orphan_user(self):
        """If the last message is a bare user prompt with no assistant reply,
        drop it. Called on resume to recover from a crashed turn."""
        if not self.messages:
            return
        last = self.messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            self.messages.pop()
            self._write({"type": "drop_orphan_user", "reason": "crash recovery"})

    def clear(self):
        self.messages.clear()
        self.todos.clear()
        self._write({"type": "clear",
                     "ts": datetime.now(timezone.utc).isoformat()})

    def replace_messages(self, new_messages: list[dict], reason: str):
        """Replace conversation state after compaction and persist the snapshot."""
        self.messages = new_messages

        self._write({
            "type": "compact",
            "reason": reason,
            "messages": new_messages,
            "len_after": len(new_messages),
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    def _write(self, entry: dict):
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()

    def _replay(self):
        for line in self.path.open():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = entry.get("type")
            if t == "user":
                self.messages.append({"role": "user", "content": entry["content"]})
            elif t == "assistant":
                self.messages.append({"role": "assistant", "content": entry["content"]})
            elif t == "tool_results":
                self.messages.append({"role": "user", "content": entry["content"]})
            elif t == "clear":
                self.messages.clear()
            elif t == "compact":
                compacted = entry.get("messages")
                if isinstance(compacted, list):
                    self.messages = compacted
            elif t == "drop_orphan_user":
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages.pop()


#8.compaction
SUMMARIZER = (
    "You are summarizing a conversation. Preserve identifiers (uuids, paths, "
    "urls, names) verbatim, active tasks, decisions, and pending TODOs. Drop "
    "redundant tool output, idle exploration, errors that were resolved. "
    "Past tense. Under 1000 words.")

def compact_messages(client: Anthropic, messages: list[dict]) -> list[dict]:
    """Summarize older messages into one synthetic user message; keep recent verbatim."""
    if len(messages) <= KEEP_RECENT + 1:
        return messages

    old, recent = messages[:-KEEP_RECENT], messages[-KEEP_RECENT:]

    # CRITICAL: the summarizer call cannot end on an assistant tool_use block —
    # the API requires tool_result to follow. Trim trailing tool_use turns.
    while old and _is_tool_use_assistant(old[-1]):
        recent.insert(0, old.pop())

    if not old:
        return messages       # nothing safe to compact

    console.print(f"  [dim]compacting {len(old)} old messages...[/dim]")

    try:
        r = client.messages.create(
            model=MODEL, max_tokens=2048, system=SUMMARIZER,
            messages=old + [{"role": "user",
                             "content": "Summarize everything above per instructions."}],
        )
    except Exception as e:
        console.print(f"  [red]compaction failed: {e}[/red]")
        return messages

    summary = "".join(b.text for b in r.content if b.type == "text")
    preamble = (
        "<conversation_summary>\nThe following is a summary of our prior "
        f"conversation. Treat as memory; continue from here.\n\n{summary}\n"
        "</conversation_summary>")
    return [{"role": "user", "content": preamble}] + recent

def _is_tool_use_assistant(msg: dict) -> bool:
    if msg["role"] != "assistant":
        return False
    content = msg["content"]
    if isinstance(content, list):
        return any((b.get("type") if isinstance(b, dict) else getattr(b, "type", None))
                   == "tool_use" for b in content)
    return False

RETRYABLE = (anthropic.RateLimitError,
             anthropic.APIConnectionError,
             anthropic.InternalServerError,
             anthropic.APITimeoutError)

def with_retries(fn, *args, max_attempts=5, base=1.0, cap=30.0, **kwargs):
    """Call fn with full-jitter exponential backoff on retryable errors."""
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except RETRYABLE as e:
            if attempt == max_attempts - 1:
                raise
            delay = min(cap, base * (2 ** attempt)) * random.random()
            console.print(f"  [yellow]API {type(e).__name__}; retry in {delay:.1f}s "
                          f"(attempt {attempt + 1}/{max_attempts})[/yellow]")
            time.sleep(delay)

#9.the streaming agent loop
DEFAULT_SYSTEM = """\
You are a coding agent that runs in the user's terminal.
You help the user inspect, modify, execute, and reason about code and files.
Be precise and concise.

You have local tools for filesystem, shell, search, and task tracking.
You may also have dynamically discovered MCP tools whose names begin with
mcp__. Use the most appropriate tool for the task.
You may also have reusable skills listed in the Skills catalog.
When a skill is relevant, call read_skill before applying its instructions.
You also have a Task tool that delegates self-contained research or
code-inspection work to a fresh read-only subagent.

Use Task when an investigation would otherwise fill the parent context
with many files or large amounts of temporary information.

Give the subagent a complete standalone task description because it
cannot see the parent's conversation.

Workflow on any non-trivial task:
  1. Gather context first (Read, Glob, Grep) — never invent file paths.
  2. For multi-step work, call TodoWrite up front to plan; update as you go.
  3. Take action (Write/Edit/Bash) — these prompt the user for permission.
  4. Verify (run tests, open the result, read the output).

Rules:
  - Read a file BEFORE editing it. Edit fails if old_string isn't unique —
    include surrounding lines for context.
  - When a command fails, read the error, try one fix, then ask the user.
  - Prefer surgical Edits over wholesale Writes.
  - Keep responses short. The user is reading the source code as they use you.
  - For multi-tool turns, batch in one assistant message — don't serialize
    when parallel works."""

def _build_system_param(system_text: str) -> str:
    return system_text

def _build_tools_param(
    allowed_tool_names: set[str] | None = None,
) -> list[dict]:
    tools = TOOLS + MCP_TOOLS

    if allowed_tool_names is None:
        return tools

    return [
        tool for tool in tools
        if tool["name"] in allowed_tool_names
    ]

def stream_one_turn(
    client: Anthropic,
    system,
    messages: list[dict],
    on_text=None,
    allowed_tool_names: set[str] | None = None,
):
    """Stream one assistant turn through a callback or console output.
    Return (assistant_blocks, stop_reason, usage)."""
    blocks_in_progress: dict[int, dict] = {}
    stop_reason = None
    usage = None

    with client.messages.stream(
        model=MODEL, max_tokens=4096,
        system=system, tools=_build_tools_param(allowed_tool_names),
        messages=messages,
    ) as stream:
        for event in stream:
            t = event.type

            if t == "content_block_start":
                cb = event.content_block

                if cb.type == "thinking":
                    blocks_in_progress[event.index] = {"type": "thinking","thinking": "","signature": ""}

                elif cb.type == "text":
                    blocks_in_progress[event.index] = {"type": "text", "text": ""}

                elif cb.type == "tool_use":
                    blocks_in_progress[event.index] = {
                        "type": "tool_use", "id": cb.id, "name": cb.name,
                        "_partial_json": "",
                    }
                    # tool_use always starts on a new line, after any prior text.
                    if on_text is None:
                        console.print()

            elif t == "content_block_delta":
                d = event.delta
                blk = blocks_in_progress.get(event.index)
                if blk is None:
                    continue
                if d.type == "thinking_delta":
                    blk["thinking"] += d.thinking

                elif d.type == "signature_delta":
                    blk["signature"] += d.signature


                elif d.type == "text_delta":
                    if on_text is None:
                        console.print(
                            d.text,
                            end="",
                            soft_wrap=True,
                            highlight=False,
                        )
                    else:
                        on_text(d.text)
                    blk["text"] += d.text
                elif d.type == "input_json_delta":
                    blk["_partial_json"] += d.partial_json

            elif t == "content_block_stop":
                blk = blocks_in_progress.get(event.index)
                if blk and blk["type"] == "thinking":
                    final_block = getattr(event, "content_block", None)
                    if final_block is not None and hasattr(final_block, "model_dump"):
                        blocks_in_progress[event.index] = final_block.model_dump()

                elif blk and blk["type"] == "tool_use":
                    raw = blk.pop("_partial_json") or "{}"
                    try:
                        blk["input"] = json.loads(raw)
                    except json.JSONDecodeError as e:
                        blk["_json_error"] = str(e)

            elif t == "message_delta":
                stop_reason = event.delta.stop_reason
                if event.usage:
                    usage = event.usage

            elif t == "message_stop":
                # the final message object holds the canonical usage record.
                final = event.message
                if getattr(final, "usage", None):
                    usage = final.usage

    blocks = [blocks_in_progress[i] for i in sorted(blocks_in_progress)]
    if on_text is None and any(
            b["type"] == "text" and b["text"].strip()
            for b in blocks
    ):
        console.print()                   # newline after the streamed text
    return blocks, stop_reason, usage

class AgentTurnError(RuntimeError):
    """An Agent turn failed to complete."""
    pass

def extract_assistant_text(content_blocks: list[dict]) -> str:
    """Extract visible text from assistant content blocks."""
    return "".join(
        block.get("text", "")
        for block in content_blocks
        if block.get("type") == "text"
    )

def agent_turn(
    client: Anthropic,
    session: Session,
    system_text: str,
    meter: Meter,
    user_input: str,
    on_text=None,
    allowed_tool_names: set[str] | None = None,
    max_turns: int = MAX_TURNS,
    raise_on_error: bool = False,
):
    if not 1 <= max_turns <= MAX_TURNS:
        raise ValueError(
            f"max_turns must be between 1 and {MAX_TURNS}"
        )

    def fail(message: str):
        if raise_on_error:
            raise AgentTurnError(message)

        console.print(f"\n[red]ERROR: {message}[/red]")
        return None

    session.append_user(user_input)
    system = _build_system_param(system_text)

    for turn in range(max_turns):
        # auto-compact when approaching context budget
        if turn > 0 and turn % 5 == 0:
            try:
                in_toks = with_retries(client.messages.count_tokens,
                                       model=MODEL, system=system, messages=session.messages,
                                       tools=_build_tools_param(allowed_tool_names)).input_tokens
                if in_toks > CONTEXT_WINDOW * COMPACT_TRIGGER:
                    session.replace_messages(
                        compact_messages(client, session.messages),
                        reason=f"auto: {in_toks} tokens")
            except Exception:
                pass     # token count is a nice-to-have; don't break the loop

        try:
            if USE_STREAM:
                blocks, stop_reason, usage = with_retries(
                    stream_one_turn,
                    client,
                    system,
                    session.messages,
                    on_text=on_text,
                    allowed_tool_names=allowed_tool_names,
                )

                content_blocks = blocks
            else:
                r = with_retries(client.messages.create,
                                 model=MODEL, max_tokens=4096,
                                 system=system, tools=_build_tools_param(allowed_tool_names),
                                 messages=session.messages)
                content_blocks = [b.model_dump() for b in r.content]
                stop_reason = r.stop_reason
                usage = r.usage
                # render for non-stream
                for b in r.content:
                    if b.type == "text":
                        render_text(b.text)
        except KeyboardInterrupt:
            console.print("\n[dim]turn cancelled. session preserved.[/dim]")
            return
        except RETRYABLE as e:
            if raise_on_error:
                raise AgentTurnError("Model API request failed after retries.") from e
            console.print(f"\n[red]API error after retries: {e}[/red]")
            return None
        except Exception as e:
            if raise_on_error:
                raise AgentTurnError("Unexpected model request failure.") from e
            console.print(f"\n[red]unexpected error: {type(e).__name__}: {e}[/red]")
            return None

        if stop_reason == "max_tokens":
            return fail("Response was truncated because max_tokens was reached.")

        bad_tool = next(
            (b for b in content_blocks if b.get("_json_error")),
            None,
        )

        if bad_tool is not None:
            return fail(
                f"Invalid JSON arguments for tool "
                f"{bad_tool['name']}: {bad_tool['_json_error']}"
            )

        session.append_assistant(content_blocks)

        if usage is not None:
            spent = meter.add(usage)
            console.print(f"  [dim]turn {turn} · ¥{spent:.4f}  total {meter.status()}[/dim]")

        if stop_reason == "tool_use":
            # Continue below and execute the requested tools.
            pass

        elif stop_reason in ("end_turn", "stop_sequence"):
            # Normal completion: return the final assistant text.
            return extract_assistant_text(content_blocks)

        elif stop_reason == "refusal":
            return fail("Model stopped with refusal.")

        elif stop_reason == "pause_turn":
            return fail("Model paused; continuation is not implemented.")


        elif stop_reason == "model_context_window_exceeded":
            return fail("Model context window was exceeded.")

        else:
            return fail(f"Unexpected stop_reason: {stop_reason!r}")

        # dispatch tools, optionally asking for permission
        results = []
        for b in content_blocks:
            if b.get("type") != "tool_use":
                continue
            name, args = b["name"], b.get("input", {})
            render_tool_call(name, args)

            if (
                    allowed_tool_names is not None
                    and name not in allowed_tool_names
            ):
                out = f"ERROR: tool not permitted: {name}"

                render_tool_result(name, out)

                results.append({
                    "type": "tool_result",
                    "tool_use_id": b["id"],
                    "content": out,
                    "is_error": True,
                })
                continue

            if name in WRITE_TOOLS and not ask_permission(name, args):
                results.append({"type": "tool_result", "tool_use_id": b["id"],
                                "content": "user denied this tool call.",
                                "is_error": True})
                continue

            try:
                if name.startswith("mcp__"):
                    server_id, tool_name = parse_mcp_tool_name(name)

                    mcp_client = MCP_CLIENTS.get(server_id)

                    if mcp_client is None:
                        out = f"ERROR: unknown MCP server: {server_id}"
                    else:
                        out = mcp_client.call_tool(tool_name, args)

                elif name == "Task":
                    description = args["description"]

                    out, child_usages = run_subagent(
                        client,
                        description,
                        model=MODEL,
                        tools=build_subagent_tools(),
                        dispatch=build_subagent_dispatch(),
                    )

                    # Subagent API usage must also count toward total cost.
                    for child_usage in child_usages:
                        meter.add(child_usage)

                else:
                    handler = DISPATCH.get(name)

                    if handler is None:
                        out = f"ERROR: unknown tool: {name}"

                    elif name == "TodoWrite":
                        out = handler(
                            session=session,
                            **args,
                        )

                    else:
                        out = handler(**args)

            except Exception as e:
                out = f"ERROR: {type(e).__name__}: {e}"

            render_tool_result(name, out)

            tool_result = {
                "type": "tool_result",
                "tool_use_id": b["id"],
                "content": out,
            }

            if str(out).startswith("ERROR:"):
                tool_result["is_error"] = True

            results.append(tool_result)

        session.append_tool_results(results)

    message = f"Agent exceeded maximum turns ({max_turns})."

    if raise_on_error:
        raise AgentTurnError(message)

    console.print(f"\n[red]ERROR: {message}[/red]")
    return None

#10.slash commands
def handle_slash(client: Anthropic, session: Session, meter: Meter,
                 cmd: str, args: str) -> bool:
    if cmd == "/help":
        console.print("[bold]commands:[/bold]")
        console.print("  /help                show this")
        console.print("  /cost                show token & yuan usage")
        console.print("  /model               show current model")
        console.print("  /init                create AGENT.md from this directory")
        console.print("  /clear               forget conversation; keep file on disk")
        console.print("  /compact             summarize old messages and continue")
        console.print("  /resume <id>         (run from shell: python agent.py --resume <id>)")
        console.print("  /exit                quit (or ctrl-d)")
        return True
    if cmd == "/cost":
        console.print(f"  [bold]cost[/bold]   {meter.status()}")
        console.print(f"  [bold]turns[/bold]  {meter.turns}")
        return True
    if cmd == "/model":
        console.print(f"  [bold]model[/bold]  {MODEL}")
        return True
    if cmd == "/init":
        path = Path("AGENT.md")
        if path.exists():
            console.print(f"  [yellow]{path} already exists; not overwriting[/yellow]")
        else:
            path.write_text(
                "# Project context\n\n"
                f"This file was generated by `/init` in {Path.cwd()}.\n\n"
                "## What this project is\n_Describe your project here._\n\n"
                "## Conventions\n- _Coding style, naming, etc._\n\n"
                "## How to test\n- _How to run tests / what passes._\n")
            console.print(f"  [green]created {path.resolve()}[/green]")
        return True
    if cmd == "/clear":
        session.clear()
        console.print("  [dim]cleared.[/dim]")
        return True
    if cmd == "/compact":
        session.replace_messages(compact_messages(client, session.messages),
                                 reason="manual")
        console.print(f"  [dim]compacted; {len(session.messages)} messages remain.[/dim]")
        return True
    if cmd == "/resume":
        console.print(f"  [dim]restart with: python agent.py --resume {args}[/dim]")
        return True
    if cmd == "/exit":
        sys.exit(0)
    return False

#11.main
def main():
    global USE_STREAM

    ap = argparse.ArgumentParser(prog="agentic-rag-service", add_help=True)
    ap.add_argument("prompt", nargs="*", help="one-shot prompt; omit for interactive REPL")
    ap.add_argument("--resume", help="resume session by id")
    ap.add_argument("--session-dir", action="store_true",
                    help="print the session directory and exit")
    ap.add_argument("--no-stream", action="store_true",
                    help="disable streaming (useful for debugging)")
    args = ap.parse_args()

    if args.session_dir:
        print(SESSION_DIR)
        return

    USE_STREAM = USE_STREAM and not args.no_stream

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("set DEEPSEEK_API_KEY first")

    client = Anthropic(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

    try:
        init_mcp_clients()
        session = Session(session_id=args.resume)
        if args.resume:
            session.truncate_orphan_user()

        meter = Meter()

        agent_md = load_agent_md(Path.cwd())
        skill_catalog = build_skill_catalog()

        system = (
                DEFAULT_SYSTEM
                + "\n\n# Skills\n"
                + skill_catalog
        )

        if agent_md:
            system += (
                    "\n\n# project context (AGENT.md)\n"
                    + agent_md
            )

        console.print(Panel.fit(
            f"[bold]agentic-rag-service[/bold]   [dim]session {session.id}[/dim]\n"
            f"[dim]model {MODEL}   cwd {Path.cwd()}   "
            f"streaming={'on' if USE_STREAM else 'off'}   "
            f"cache=deepseek-auto[/dim]",
            border_style="dim"
        ))

        if args.prompt:
            agent_turn(
                client,
                session,
                system,
                meter,
                " ".join(args.prompt),
            )
            console.print(f"\n[dim]{meter.status()}[/dim]")
            return

        console.print(
            "[dim]/help for commands · ctrl-d to exit[/dim]\n"
        )

        while True:
            try:
                line = console.input("[bold blue]>[/bold blue] ")
            except (EOFError, KeyboardInterrupt):
                print()
                console.print(
                    f"[dim]final: {meter.status()}[/dim]"
                )
                return

            if not line.strip():
                continue

            if line.startswith("/"):
                cmd, _, rest = line.partition(" ")

                if handle_slash(
                        client,
                        session,
                        meter,
                        cmd,
                        rest,
                ):
                    continue

            try:
                agent_turn(
                    client,
                    session,
                    system,
                    meter,
                    line,
                )
            except KeyboardInterrupt:
                console.print(
                    "\n[dim]cancelled. session preserved.[/dim]"
                )

    finally:
        close_mcp_clients()

if __name__ == "__main__":
    main()
