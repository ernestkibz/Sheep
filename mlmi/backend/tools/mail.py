"""Mail and calendar over open standards only: IMAP and SMTP for mail, .ics
(iCalendar, RFC 5545) for the calendar. Sheep talks to any mail server the
person already has; no outside product sits in between (plan A10, 2026-10-07).

The account is read from HOME/mail.json, or from the environment:
  SHEEP_MAIL_IMAP, SHEEP_MAIL_SMTP   host[:port]
  SHEEP_MAIL_USER, SHEEP_MAIL_PASSWORD, SHEEP_MAIL_FROM
  SHEEP_MAIL_SSL   "0" for a plain connection (a local test server)

Sending always asks the person first (confirm="always"); a draft is an .eml
file in the workspace, so it can be read, changed and undone like any file.
"""
import email
import email.policy
import imaplib
import json
import os
import re
import smtplib
import time
import uuid
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr

from . import tool, overwrite_check
from ..paths import HOME
from ..workspace import ToolError

CALENDAR = "calendar.ics"
LIST_MAX = 20


def account():
    """The mail account: mail.json, then the environment."""
    cfg = {}
    f = HOME / "mail.json"
    if f.exists():
        try:
            cfg = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            raise ToolError(f"{f} is not valid JSON")
    env = lambda k: os.environ.get("SHEEP_MAIL_" + k.upper(), "")
    out = {k: cfg.get(k) or env(k) for k in ("imap", "smtp", "user", "password", "from")}
    ssl = cfg.get("ssl", env("ssl") or "1")
    out["ssl"] = str(ssl).lower() not in ("0", "false", "no", "off")
    return out


def _host(spec, ssl, imap):
    host, _, port = str(spec or "").partition(":")
    if not host:
        raise ToolError("no mail server is set up yet. Add it in ~/.mlmi/mail.json "
                        '({"imap": "imap.example.com", "smtp": "smtp.example.com", "user": "...", "password": "..."})')
    default = (993 if ssl else 143) if imap else (465 if ssl else 25)
    return host, int(port) if port.isdigit() else default


def _imap(acc):
    host, port = _host(acc["imap"], acc["ssl"], True)
    try:
        m = (imaplib.IMAP4_SSL if acc["ssl"] else imaplib.IMAP4)(host, port)
        if acc["user"]:
            m.login(acc["user"], acc["password"] or "")
        return m
    except (OSError, imaplib.IMAP4.error) as e:
        raise ToolError(f"could not reach the mail server {host}: {e}")


def _bye(m):
    try:
        m.logout()
    except Exception:
        pass


def _text_of(msg):
    """The plain text of a message, or its HTML with the tags taken out."""
    part = msg.get_body(preferencelist=("plain", "html")) or msg
    try:
        body = part.get_content()
    except (KeyError, LookupError):
        body = str(part.get_payload(decode=True) or b"", "utf-8", "replace")
    if part.get_content_type() == "text/html":
        body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"[ \t]+", " ", str(body)).strip()


@tool("mail_list", "mail", 1, "List the newest emails in the inbox (or another folder): number, sender, subject, date.",
      {"folder": ("string", "mail folder, default INBOX"), "count": ("integer", "how many, default 10"),
       "search": ("string", "only emails whose subject or sender has these words, optional")})
def mail_list(ctx, folder="INBOX", count=10, search=""):
    m = _imap(account())
    try:
        typ, _ = m.select(folder or "INBOX", readonly=True)
        if typ != "OK":
            raise ToolError(f"there is no mail folder called {folder}")
        q = str(search).replace('"', "")
        crit = ["ALL"] if not q else ["OR", "SUBJECT", f'"{q}"', "FROM", f'"{q}"']
        typ, data = m.search(None, *crit)
        ids = (data[0] or b"").split()[-max(1, min(int(count or 10), LIST_MAX)):]
        lines = []
        for i in reversed(ids):
            typ, got = m.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            raw = next((g[1] for g in got if isinstance(g, tuple)), b"")
            h = email.message_from_bytes(raw, policy=email.policy.default)
            lines.append(f"#{i.decode()}  {h.get('From', '')}  |  {h.get('Subject', '(no subject)')}  |  {h.get('Date', '')}")
        return "\n".join(lines) if lines else "no emails" + (f" matching '{q}'" if q else "")
    finally:
        _bye(m)


@tool("mail_read", "mail", 1, "Read one email by its number from mail_list.",
      {"number": ("integer", "the email's number"), "folder": ("string", "mail folder, default INBOX")}, ("number",))
def mail_read(ctx, number, folder="INBOX"):
    m = _imap(account())
    try:
        m.select(folder or "INBOX", readonly=True)
        typ, got = m.fetch(str(int(number)).encode(), "(BODY.PEEK[])")
        raw = next((g[1] for g in got or [] if isinstance(g, tuple)), None)
        if typ != "OK" or not raw:
            raise ToolError(f"there is no email #{number} in {folder}")
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        return (f"From: {msg.get('From', '')}\nTo: {msg.get('To', '')}\nDate: {msg.get('Date', '')}\n"
                f"Subject: {msg.get('Subject', '')}\n\n{_text_of(msg)[:6000]}")
    finally:
        _bye(m)


def _message(to, subject, body):
    if not to or "@" not in parseaddr(str(to).split(",")[0])[1]:
        raise ToolError("an email needs a 'to' address such as name@example.com")
    acc = account()
    msg = EmailMessage()
    msg["From"] = acc["from"] or acc["user"] or "me@localhost"
    msg["To"], msg["Subject"] = to, subject or ""
    msg["Date"], msg["Message-ID"] = formatdate(localtime=True), make_msgid(domain="sheep.local")
    msg.set_content(body or "")
    return msg


@tool("mail_draft", "mail", 2, "Write an email as a draft (.eml in the workspace) without sending it.",
      {"to": ("string", "who it is for"), "subject": ("string", "subject line"), "body": ("string", "the email's text"),
       "path": ("string", "draft file name, optional")}, ("to", "subject", "body"), confirm=overwrite_check("path"))
def mail_draft(ctx, to, subject, body, path=""):
    msg = _message(to, subject, body)
    slug = re.sub(r"[^a-z0-9]+", "_", (subject or "draft").lower()).strip("_")[:40] or "draft"
    p = ctx.ws.path(path or f"drafts/{slug}.eml")
    with ctx.ws.changing(p, by=ctx.by, what=f"mail draft {ctx.ws.rel(p)}"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes(msg))
    ctx.made.add(ctx.ws.rel(p))
    return f"draft saved as {ctx.ws.rel(p)} (not sent; mail_send sends it after the person agrees)"


@tool("mail_send", "mail", 2, "Send an email over SMTP: a draft file, or to/subject/body. Always asks the person first.",
      {"draft": ("string", "a .eml draft in the workspace, optional"), "to": ("string", "who it is for"),
       "subject": ("string", "subject line"), "body": ("string", "the email's text")}, confirm="always")
def mail_send(ctx, draft="", to="", subject="", body=""):
    if draft:
        msg = email.message_from_bytes(ctx.ws.path(draft, must_exist=True).read_bytes(), policy=email.policy.default)
        if not msg["To"]:
            raise ToolError(f"{draft} has no 'To' address")
    else:
        msg = _message(to, subject, body)
    acc = account()
    host, port = _host(acc["smtp"], acc["ssl"], False)
    try:
        s = smtplib.SMTP_SSL(host, port, timeout=30) if acc["ssl"] and port == 465 else smtplib.SMTP(host, port, timeout=30)
        with s:
            if acc["ssl"] and port != 465:
                s.starttls()
            if acc["user"] and acc["password"]:
                s.login(acc["user"], acc["password"])
            refused = s.send_message(msg)
    except (OSError, smtplib.SMTPException) as e:
        raise ToolError(f"the email was not sent: {e}")
    if refused:
        raise ToolError(f"the server refused: {', '.join(refused)}")
    return f"sent to {msg['To']}: {msg['Subject']}"



# ------------------------------------------------------------------ calendar
def _when(s, name):
    s = str(s or "").strip().replace("T", " ")
    for f in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    raise ToolError(f"{name} must be a date and time like 2026-10-30 14:00")


def _esc(t):
    return str(t or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _unesc(t):
    return t.replace("\\n", "\n").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")


def events(text):
    """The events of an .ics text: [{summary, start, end, location, uid, description}]."""
    text = re.sub(r"\r?\n[ \t]", "", text)          # unfold continued lines
    out, cur = [], None
    for line in text.splitlines():
        if line == "BEGIN:VEVENT":
            cur = {}
        elif line == "END:VEVENT" and cur is not None:
            out.append(cur)
            cur = None
        elif cur is not None and ":" in line:
            key, val = line.split(":", 1)
            key = key.split(";")[0].upper()
            if key in ("DTSTART", "DTEND"):
                v = val.rstrip("Z")
                try:
                    cur["start" if key == "DTSTART" else "end"] = datetime.strptime(v, "%Y%m%dT%H%M%S" if "T" in v else "%Y%m%d")
                except ValueError:
                    pass
            elif key in ("SUMMARY", "LOCATION", "UID", "DESCRIPTION"):
                cur[key.lower()] = _unesc(val)
    return out


@tool("calendar_add", "mail", 2, "Add an event to the calendar (calendar.ics, opens in any calendar app).",
      {"title": ("string", "what the event is"), "start": ("string", "when it starts, e.g. 2026-10-30 14:00"),
       "minutes": ("integer", "how long, default 60"), "location": ("string", "where, optional"),
       "notes": ("string", "details, optional"), "path": ("string", "calendar file, default calendar.ics")},
      ("title", "start"))
def calendar_add(ctx, title, start, minutes=60, location="", notes="", path=CALENDAR):
    t0 = _when(start, "start")
    # a 3B once sent 720 for "remind me ... at 9" (2026-10-19): a reminder is short, and a
    # half-day event nobody described is a slip, so the model is asked again rather than obeyed
    if int(minutes or 60) > 8 * 60:
        raise ToolError(f"{minutes} minutes is {int(minutes) / 60:.0f} hours; for a reminder or a call use "
                        "minutes 15, for a meeting 30-60; give more only if the person said how long")
    t1 = t0 + timedelta(minutes=max(1, int(minutes or 60)))
    p = ctx.ws.path(path or CALENDAR)
    if p.suffix.lower() != ".ics":
        raise ToolError("a calendar is an .ics file")
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    if old and "END:VCALENDAR" not in old:
        raise ToolError(f"{path} is not a calendar file")
    fmt = "%Y%m%dT%H%M%S"
    ev = ["BEGIN:VEVENT", f"UID:{uuid.uuid4()}@sheep", f"DTSTAMP:{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
          f"DTSTART:{t0.strftime(fmt)}", f"DTEND:{t1.strftime(fmt)}", f"SUMMARY:{_esc(title)}"]
    ev += [f"LOCATION:{_esc(location)}"] if location else []
    ev += [f"DESCRIPTION:{_esc(notes)}"] if notes else []
    ev.append("END:VEVENT")
    if old:
        head = old[:old.rindex("END:VCALENDAR")].rstrip("\r\n")
        text = head + "\r\n" + "\r\n".join(ev) + "\r\nEND:VCALENDAR\r\n"
    else:
        text = "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Sheep//Calendar//EN"] + ev + ["END:VCALENDAR"]) + "\r\n"
    with ctx.ws.changing(p, by=ctx.by, what=f"calendar add {title}"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
    ctx.made.add(ctx.ws.rel(p))
    return f"added '{title}' on {t0:%a %d %b %Y, %H:%M}-{t1:%H:%M} to {ctx.ws.rel(p)}"


@tool("calendar_list", "mail", 1, "List the events in a calendar (.ics), soonest first, optionally from a date.",
      {"path": ("string", "calendar file, default calendar.ics"), "from_date": ("string", "only events from this date, optional")})
def calendar_list(ctx, path=CALENDAR, from_date=""):
    p = ctx.ws.path(path or CALENDAR)
    if not p.exists():
        return "the calendar is empty"
    evs = [e for e in events(p.read_text(encoding="utf-8", errors="replace")) if "start" in e]
    if from_date:
        since = _when(from_date, "from_date")
        evs = [e for e in evs if e["start"] >= since]
    evs.sort(key=lambda e: e["start"])
    if not evs:
        return "no events" + (f" from {from_date}" if from_date else "")
    return "\n".join(f"{e['start']:%a %d %b %Y %H:%M}  {e.get('summary', '(no title)')}"
                     + (f"  @ {e['location']}" if e.get("location") else "") for e in evs[:50])

