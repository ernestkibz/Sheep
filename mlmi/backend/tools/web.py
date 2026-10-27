"""ABIS web access: search the web and read pages, for any model through the
orchestrator.

Search uses Tavily when TAVILY_API_KEY is set (an AI-search API built for this:
ranked results with clean content), and a keyless public search otherwise, so
it works on every machine. Pages are read only from public HTTPS hosts: the
address and every redirect are resolved and refused if they point inside the
machine or the local network. Every query and page is written to the "Sent to
the web" log, and the person can switch web access off in Settings.
"""
import base64
import html
import ipaddress
import json
import os
import re
import socket
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from . import tool
from ..workspace import ToolError

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Sheep/1.0"
# Wikimedia's user-agent policy: a browser-like agent got 429 "try again later" on every search,
# one that names the program and where it lives is answered (2026-10-21)
WIKI_UA = "Sheep/1.0 (https://github.com/ernestkibz/Sheep; desktop assistant)"
MAX_PAGE = 2_000_000
SENT = []                       # the "Sent to the web" log: {t, kind, what, via}
STATE = {"on": True}            # the Settings switch


def _log(kind, what, via):
    SENT.append({"t": time.time(), "kind": kind, "what": str(what)[:300], "via": via})
    del SENT[:-500]


def _public_https(raw):
    """A public HTTPS address, or a ToolError."""
    try:
        u = urllib.parse.urlsplit(str(raw).strip())
        if u.scheme.lower() != "https" or not u.hostname or u.username or u.password:
            raise ValueError
        host = u.hostname.rstrip(".").lower()
        if host in ("localhost",) or host.endswith((".local", ".internal", ".lan", ".home")):
            raise ValueError
        try:
            addrs = [ipaddress.ip_address(host)]
        except ValueError:
            addrs = [ipaddress.ip_address(i[4][0]) for i in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)]
        if not addrs or any(not a.is_global for a in addrs):
            raise ValueError
        return urllib.parse.urlunsplit(("https", u.netloc, u.path or "/", u.query, ""))
    except (ValueError, OSError, UnicodeError):
        raise ToolError("pages can be read only from public https:// addresses")


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_Redirects)


def _get(url, data=None, headers=None, timeout=20):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    with _OPENER.open(req, timeout=timeout) as r:
        raw = r.read(MAX_PAGE + 1)
        ctype = r.headers.get("Content-Type", "")
    if len(raw) > MAX_PAGE:
        raw = raw[:MAX_PAGE]
    cs = re.search(r"charset=([\w-]+)", ctype)
    return raw.decode(cs.group(1) if cs else "utf-8", errors="replace"), ctype


class _Text(HTMLParser):
    """Readable text of a page: no scripts, styles, navigation or forms."""
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "form", "svg", "aside", "iframe"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)

    def text(self):
        t = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.out))
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def search(query, n=5):
    """[{title, url, snippet}] for a query."""
    if not STATE["on"]:
        raise ToolError("web access is off in Settings")
    key = os.environ.get("TAVILY_API_KEY", "")
    if key:
        body = json.dumps({"query": query, "max_results": n, "search_depth": "basic"}).encode()
        raw, _ = _get("https://api.tavily.com/search", data=body,
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        _log("search", query, "Tavily")
        res = json.loads(raw).get("results") or []
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("content") or "")[:400]}
                for r in res[:n]]
    # keyless (DuckDuckGo answers scripts with a CAPTCHA, 2026-10-05): Bing News
    # for news-like questions, Bing's web results that share the question's key
    # words (to a cookieless client Bing gave the same "Uganda" pages for any
    # Uganda query), then Wikipedia's search API, which never refuses
    out, via = [], []
    kw = keywords(query)
    if NEWSY.search(query):
        try:
            out = _bing_news(" ".join(kw) or query, n)
            if out:
                via.append("Bing News")
        except Exception:
            pass
    if len(out) < n:
        try:
            more = [r for r in _bing(query, n * 2) if relevant(r, kw) and r["url"] not in {x["url"] for x in out}]
            if more:
                via.append("Bing")
            out += more[:n - len(out)]
        except Exception:
            pass
    if not out:
        # Wikipedia matches titles: the question's names ("Mount Elgon") beat its words ("tall")
        names = [m for m in re.findall(r"(?<!^)(?<![.?!] )\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", query.strip())
                 if m.lower() not in STOP]
        out = _wiki_retry(" ".join(names), n) if names else []
        out = out or _wiki_retry(" ".join(kw) or query, n)
        via.append("Wikipedia")
    _log("search", query, ", ".join(via) or "none")
    return out


def _wiki_retry(q, n, wait=2.0):
    """Wikipedia, asked once more after a busy answer: it said 429 "try again later" after a
    morning of test runs and the search failed outright (B10 replay, 2026-10-21)."""
    import time
    try:
        return _wikipedia(q, n)
    except Exception as e:
        if "429" not in str(e) and "503" not in str(e):
            raise
        time.sleep(wait)
        return _wikipedia(q, n)


NEWSY = re.compile(r"\b(news|latest|today|tonight|yesterday|this (week|month)|score|result|match|won|win|election|"
                   r"breaking|update|recent(ly)?|announced?)\b", re.I)
STOP = set("a an the of in on at to for from by with about and or is are was were be been what which who whom whose "
           "when where why how do does did can could should would will shall may might latest news today current "
           "currently recent recently tell me please give show find search look up any some this that these those "
           "it its their there here my your our week month year now right".split())


def keywords(query):
    return [w for w in re.findall(r"[\w'-]+", query.lower()) if w not in STOP and len(w) > 1]


def relevant(r, kw):
    """At least half the question's key words appear in the result's title or snippet."""
    if not kw:
        return True
    hay = (r["title"] + " " + r["snippet"]).lower()
    return sum(w in hay for w in kw) * 2 >= len(kw)


def _bing_news(query, n):
    raw, _ = _get("https://www.bing.com/news/search?" + urllib.parse.urlencode({"q": query, "format": "rss"}),
                  headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                         "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"})
    out = []
    for it in re.findall(r"<item>(.*?)</item>", raw, re.S):
        t = re.search(r"<title>(.*?)</title>", it, re.S)
        lk = re.search(r"<link>(.*?)</link>", it, re.S)
        d = re.search(r"<description>(.*?)</description>", it, re.S)
        when = re.search(r"<pubDate>(.*?)</pubDate>", it)
        if not t or not lk:
            continue
        link = html.unescape(lk.group(1)).strip()
        real = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query).get("url", [""])[0] or link
        if not real.startswith("https://"):
            continue
        snippet = _clean(html.unescape(d.group(1)) if d else "")
        if when:
            snippet = f"({when.group(1)[:16]}) {snippet}"
        out.append({"title": _clean(html.unescape(t.group(1)))[:200], "url": real, "snippet": snippet[:400]})
        if len(out) >= n:
            break
    return out


def _clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _bing(query, n):
    raw, _ = _get("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query}),
                  headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                         "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"})
    out = []
    for b in raw.split('class="b_algo"')[1:]:
        b = re.sub(r"<link[^>]*>|<style.*?</style>|<script.*?</script>", "", b, flags=re.S)
        a = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
        if not a:
            continue
        href = html.unescape(a.group(1))
        u = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query).get("u", [""])[0]
        if u.startswith("a1"):       # Bing's click redirect carries the address in base64
            s = u[2:]
            href = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "replace")
        if not href.startswith("https://"):
            continue
        p = re.search(r"<p[^>]*>(.*?)</p>", b, re.S)
        out.append({"title": _clean(a.group(2))[:200], "url": href, "snippet": _clean(p.group(1) if p else "")[:400]})
        if len(out) >= n:
            break
    return out


def _wikipedia(query, n):
    raw, _ = _get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": n}),
        headers={"User-Agent": WIKI_UA})
    res = (json.loads(raw).get("query") or {}).get("search") or []
    return [{"title": r["title"], "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
             "snippet": _clean(r.get("snippet", ""))[:400]} for r in res]


def fetch(url, limit=6000):
    """(title, readable text) of a public page."""
    if not STATE["on"]:
        raise ToolError("web access is off in Settings")
    url = _public_https(url)
    raw, ctype = _get(url)
    _log("page", url, "direct")
    if "html" not in ctype and not raw.lstrip().startswith("<"):
        return url, raw[:limit]
    p = _Text()
    p.feed(raw)
    return (p.title.strip() or url)[:200], p.text()[:limit]


@tool("web_search", "web", 1, "Search the web for current or unknown facts. Returns numbered sources: cite them as [1], [2].",
      {"query": ("string", "what to search for")}, ("query",))
def web_search(ctx, query):
    try:
        res = search(query)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"the search did not answer ({str(e)[:100]})")
    if not res:
        return "no results"
    srcs = getattr(ctx, "sources", None)
    if srcs is None:
        srcs = ctx.sources = []
    lines = []
    for r in res:
        if r["url"] not in [s["url"] for s in srcs]:
            srcs.append(r)
        i = [s["url"] for s in srcs].index(r["url"]) + 1
        lines.append(f"[{i}] {r['title']} ({r['url']}): {r['snippet']}")
    return "\n".join(lines)


@tool("web_fetch", "web", 1, "Read a web page from a search result for more detail.",
      {"url": ("string", "the https:// address")}, ("url",))
def web_fetch(ctx, url):
    try:
        title, text = fetch(url)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"the page did not open ({str(e)[:100]})")
    srcs = getattr(ctx, "sources", None)
    if srcs is None:
        srcs = ctx.sources = []
    if url not in [s["url"] for s in srcs]:
        srcs.append({"title": title, "url": url, "snippet": text[:200]})
    i = [s["url"] for s in srcs].index(url) + 1
    return f"[{i}] {title}\n{text}"
