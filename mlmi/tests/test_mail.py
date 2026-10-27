"""Mail and calendar over open standards (plan A10), against a local SMTP
server, a local IMAP server and an .ics file. Nothing leaves the machine.

    python -m mlmi.tests.test_mail
"""
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
import warnings
from email.message import EmailMessage
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("CLI integration test; run python -m mlmi.tests.test_mail")

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["MLMI_HOME"] = tempfile.mkdtemp(prefix="mlmi-mail-home-")
from mlmi.backend import agent  # noqa: E402
from mlmi.backend import tools as T  # noqa: E402
from mlmi.backend.workspace import ToolError, Workspace  # noqa: E402

fails, ran = [], [0]


def check(name, ok, got=""):
    ran[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      got {got}"))
    if not ok:
        fails.append(name)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _msg(i, frm, subj, body):
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = frm, "me@localhost", subj
    m["Date"] = f"Thu, 0{i} Oct 2026 09:00:00 +0300"
    m.set_content(body)
    return bytes(m)


INBOX = [_msg(1, "Ruth <ruth@example.com>", "Hackathon video", "The video must be under three minutes."),
         _msg(2, "Boaz <boaz@example.com>", "Lunch on Friday", "Shall we meet at noon?")]


class Imap(socketserver.StreamRequestHandler):
    """Just the IMAP4 commands the tools use."""
    def out(self, s):
        self.wfile.write(s if isinstance(s, bytes) else s.encode())

    def handle(self):
        self.out("* OK test IMAP4rev1 ready\r\n")
        while True:
            line = self.rfile.readline().decode(errors="replace").strip()
            if not line:
                return
            parts = line.split(" ", 2) + ["", ""]
            tag, cmd, arg = parts[0], parts[1].upper(), parts[2]
            if cmd == "CAPABILITY":
                self.out(f"* CAPABILITY IMAP4rev1\r\n{tag} OK done\r\n")
            elif cmd == "LOGIN":
                self.out(f"{tag} OK logged in\r\n")
            elif cmd in ("SELECT", "EXAMINE"):
                ok = arg.strip('"').upper() == "INBOX"
                self.out(f"* {len(INBOX)} EXISTS\r\n{tag} OK [READ-ONLY] done\r\n" if ok else f"{tag} NO no folder\r\n")
            elif cmd == "SEARCH":
                words = re.findall(r'"([^"]*)"', arg)
                hits = [str(i + 1) for i, raw in enumerate(INBOX)
                        if not words or any(w.lower() in raw.decode().lower() for w in words)]
                self.out(f"* SEARCH {' '.join(hits)}\r\n{tag} OK done\r\n")
            elif cmd == "FETCH":
                n = int(arg.split(" ")[0])
                if not 1 <= n <= len(INBOX):
                    self.out(f"{tag} OK done\r\n")
                    continue
                raw = INBOX[n - 1]
                if "HEADER.FIELDS" in arg:
                    raw = raw.split(b"\n\n", 1)[0] + b"\r\n\r\n"
                self.out(f"* {n} FETCH (BODY[] {{{len(raw)}}}\r\n".encode() + raw + f")\r\n{tag} OK done\r\n".encode())
            elif cmd == "LOGOUT":
                self.out(f"* BYE\r\n{tag} OK bye\r\n")
                return
            else:
                self.out(f"{tag} BAD unknown\r\n")


SENT = []


class Smtp(socketserver.StreamRequestHandler):
    """Just the SMTP commands smtplib sends (smtpd and asyncore left Python in 3.12)."""
    def out(self, s):
        self.wfile.write(s.encode())

    def handle(self):
        self.out("220 test SMTP ready\r\n")
        frm, to = "", []
        while True:
            line = self.rfile.readline().decode(errors="replace").rstrip("\r\n")
            if not line:
                return
            cmd = line[:4].upper()
            if cmd in ("EHLO", "HELO"):
                self.out("250 localhost\r\n")
            elif cmd == "AUTH":
                self.out("235 ok\r\n")
            elif cmd == "MAIL":
                frm, to = line.split(":", 1)[1].strip().split(" ")[0].strip("<>"), []
                self.out("250 ok\r\n")
            elif cmd == "RCPT":
                to.append(line.split(":", 1)[1].strip().strip("<>"))
                self.out("250 ok\r\n")
            elif cmd == "DATA":
                self.out("354 go ahead\r\n")
                body = []
                while True:
                    ln = self.rfile.readline()
                    if ln in (b".\r\n", b".\n", b""):
                        break
                    body.append(ln[1:] if ln.startswith(b"..") else ln)
                SENT.append((frm, to, b"".join(body)))
                self.out("250 queued\r\n")
            elif cmd == "QUIT":
                self.out("221 bye\r\n")
                return
            else:
                self.out("250 ok\r\n")


imap_port, smtp_port = free_port(), free_port()
srv = socketserver.ThreadingTCPServer(("127.0.0.1", imap_port), Imap)
srv.daemon_threads = True
threading.Thread(target=srv.serve_forever, daemon=True).start()
smtp = socketserver.ThreadingTCPServer(("127.0.0.1", smtp_port), Smtp)
smtp.daemon_threads = True
threading.Thread(target=smtp.serve_forever, daemon=True).start()
os.environ.update({"SHEEP_MAIL_IMAP": f"127.0.0.1:{imap_port}", "SHEEP_MAIL_SMTP": f"127.0.0.1:{smtp_port}",
                   "SHEEP_MAIL_USER": "me@localhost", "SHEEP_MAIL_PASSWORD": "", "SHEEP_MAIL_SSL": "0"})

ws = Workspace(Path(tempfile.mkdtemp(prefix="mlmi-mail-")))
run = lambda n, **a: T.run(ws, n, a, by="test")

# ---- read
out = run("mail_list")
check("the inbox lists both emails, newest first", "Lunch" in out and out.index("Lunch") < out.index("Hackathon"), out)
out = run("mail_list", search="hackathon")
check("a search finds only the matching email", "Hackathon" in out and "Lunch" not in out, out)
out = run("mail_read", number=1)
check("an email is read with its text", "under three minutes" in out and "ruth@example.com" in out, out)
try:
    run("mail_read", number=9)
    check("a missing email is refused", False, "no error")
except ToolError as e:
    check("a missing email is refused", "no email #9" in str(e), e)

# ---- draft, then send only after asking
out = run("mail_draft", to="ruth@example.com", subject="Re: Hackathon video", body="It will be 2 min 40 s.")
draft = out.split("draft saved as ")[1].split(" ")[0]
check("a draft is an .eml file in the workspace", (ws.root / draft).exists() and draft.endswith(".eml"), out)
check("drafting sends nothing", SENT == [], SENT)
check("sending always asks first", T.needs_confirm(ws, "mail_send", {"draft": draft}) is not None)
out = run("mail_send", draft=draft)
for _ in range(50):
    if SENT:
        break
    time.sleep(0.1)
check("the agreed email reaches the SMTP server",
      bool(SENT) and SENT[0][1] == ["ruth@example.com"] and b"2 min 40 s" in SENT[0][2], (out, SENT))
try:
    run("mail_send", to="not an address", subject="x", body="y")
    check("a bad address is refused", False, "sent")
except ToolError as e:
    check("a bad address is refused", "'to' address" in str(e), e)

# ---- calendar
out = run("calendar_add", title="Submit to Devpost", start="2026-10-28 10:00", minutes=30, location="Kampala, home")
check("an event is added", "Submit to Devpost" in out and (ws.root / "calendar.ics").exists(), out)
run("calendar_add", title="Record the video", start="2026-10-20 15:00")
ics = (ws.root / "calendar.ics").read_text(encoding="utf-8")
check("one VCALENDAR with two events", ics.startswith("BEGIN:VCALENDAR") and ics.count("BEGIN:VEVENT") == 2
      and ics.rstrip().endswith("END:VCALENDAR"), ics)
check("commas escaped per RFC 5545", "LOCATION:Kampala\\, home" in ics, ics)
out = run("calendar_list")
check("events list soonest first, with place", out.index("Record") < out.index("Submit") and "@ Kampala, home" in out, out)
out = run("calendar_list", from_date="2026-10-25")
check("from a date shows only later events", "Submit" in out and "Record" not in out, out)
ws.undo()
check("undo takes the last event off", (ws.root / "calendar.ics").read_text(encoding="utf-8").count("BEGIN:VEVENT") == 1)

# ---- the agent offers the tools when asked
names = agent.route_tools(set(T.TOOLS), "", "check my inbox for emails from ruth")
check("an inbox request gets the mail tools", {"mail_list", "mail_read"} <= set(names), names)
names = agent.route_tools(set(T.TOOLS), "", "send an email to ruth about the video")
check("a send request gets draft and send", {"mail_draft", "mail_send"} <= set(names), names)
names = agent.route_tools(set(T.TOOLS), "words", "add a meeting with Boaz on friday to my calendar")
check("a calendar request gets calendar_add inside an app", "calendar_add" in names, names)

srv.shutdown()
print(f"\n{'all ' + str(ran[0]) + ' checks passed' if not fails else str(len(fails)) + ' failed: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)

