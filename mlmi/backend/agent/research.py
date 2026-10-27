"""Web research: search, read the best pages, answer or write with sources.

Split out of agent.py (2026-10-21); every name is re-exported from the agent package."""
import uuid
from .prompts import RESEARCH
from .scope import error_words


def gather(question, emit, pages=2):
    """Search (Tavily when its key is set) and read the best pages: (results, the numbered
    sources as text for whoever writes from them)."""
    from ..tools import web
    cid = "web_" + uuid.uuid4().hex[:6]
    emit({"type": "thinking", "route": "sheep", "what": "Searching the web"})
    emit({"type": "tool_call", "id": cid, "name": "web_search", "args": {"query": question}, "route": "sheep"})
    try:
        res = web.search(question, 5)
    except Exception as e:
        res = []
        emit({"type": "tool_result", "id": cid, "name": "web_search", "ok": False, "text": str(e), "route": "sheep"})
    else:
        emit({"type": "tool_result", "id": cid, "name": "web_search", "ok": bool(res),
              "text": f"{len(res)} results" if res else "no results", "route": "sheep"})
    texts = {}
    for r in res[:pages]:
        fid = "web_" + uuid.uuid4().hex[:6]
        emit({"type": "tool_call", "id": fid, "name": "web_fetch", "args": {"url": r["url"]}, "route": "sheep"})
        try:
            _, texts[r["url"]] = web.fetch(r["url"], limit=1800)
            emit({"type": "tool_result", "id": fid, "name": "web_fetch", "ok": True, "text": "read", "route": "sheep"})
        except Exception as e:
            emit({"type": "tool_result", "id": fid, "name": "web_fetch", "ok": False, "text": str(e), "route": "sheep"})
    ctx = "\n\n".join(f"[{i + 1}] {r['title']} ({r['url']})\n{texts.get(r['url']) or r['snippet']}"
                      for i, r in enumerate(res))
    return res, ctx


def sources_md(res):
    """The closing slide or section of a file written from the web: where the facts came from."""
    return "\n\n## Sources\n" + "\n".join(f"- {r.get('title') or r['url']}: {r['url']}" for r in res[:5]) + "\n"


def research(question, convo, executor, emit, pages=2):
    """Search, read the best pages, answer with citations. Returns when done."""
    res, ctx = gather(question, emit, pages)
    if res:
        ask = f"Sources:\n{ctx}\n\nQuestion: {question}"
    else:
        ask = (f"{question}\n\n(The web search returned nothing. Answer from what you know and say plainly that "
               f"it could not be checked online.)")
    history = [m for m in convo[1:-1] if m.get("role") in ("user", "assistant") and m.get("content")][-4:]
    msgs = [{"role": "system", "content": RESEARCH}] + history + [{"role": "user", "content": ask}]
    emit({"type": "thinking", "route": "sheep", "what": f"Reading {len(res)} source{'s' if len(res) != 1 else ''}"})
    try:
        quiet = {"verify": False} if "verify" in executor.chat.__code__.co_varnames else {}
        if "on_delta" in executor.chat.__code__.co_varnames:     # the answer appears as it is written
            quiet["on_delta"] = lambda piece: emit({"type": "draft", "delta": piece})
        if getattr(executor, "retry_device", None):
            quiet["on_retry"] = lambda why: emit({"type": "device_retry", "why": why})
        msg, info = executor.chat(msgs, None, max_tokens=700, **quiet)
        text = (msg.get("content") or "").strip()
        emit({"type": "text", "text": text or "I could not write an answer from the sources.", "route": info.get("route", "local")})
    except Exception as e:
        emit({"type": "error", "text": error_words(e, local=getattr(executor, "name", "") != "nebius")})
    if res:
        emit({"type": "sources", "items": res})
    emit({"type": "done", "stopped": False, "changed": []})
