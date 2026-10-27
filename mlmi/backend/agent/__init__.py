"""The agent loop: a model, the tools it has earned, and the workspace.

Like Claude Code with VS Code: the model calls a tool, reads what happened,
and carries on until the job is done or it is stopped. Every step is an event
the interface draws live -- the call, its result, the file that changed, where
the step ran -- and nothing irreversible happens without the person's say:
deletes and overwrites wait for a yes (Auto-edit), or every write does (Ask)."""
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from .. import tools as T
from .. import checker, cues, shepherd, verify
from ..modes import LLMError, needs_plan
from ..workspace import ModelSelectionRequired, ToolError

from .prompts import (  # noqa: F401
    MAX_STEPS,
    DONE_HINT,
    fit,
    too_long,
    SYSTEM,
    ANSWER,
    CAN,
    brief,
    taught,
    answer_prompt,
    PLAN_ONLY,
    PLANNER,
    REVIEW,
    review_verdict,
    RESEARCH,
)
from .session import (  # noqa: F401
    Session,
    wait_for_model_selection,
)
from .plan_calls import (  # noqa: F401
    PLANNER_CALLS,
    _plan_calls,
    _FENCE,
    _FUNC_TAG,
    _looks_like_call,
    _call,
    _repair_call,
    _args,
)
from .routing import (  # noqa: F401
    CORE,
    WEB_CUES,
    AI_VIDEO,
    PHOTO_MONTAGE,
    VIDEO_TEXT_ONLY,
    PHOTO_ANIMATION,
    STUDIO_WORK,
    STUDIO_EDIT,
    SEE_ASK,
    WANT_PICTURE,
    MEDIA_NEEDS,
    DECK_PICTURES,
    MEDIA_KIND,
    EXTRAS,
    FILE_ASK,
    ACTION,
    CODE,
    ABILITY_Q,
    SPECIFIC,
    ability_question,
    ARTIFACT_NOUNS,
    wants_tools,
    route_tools,
    IMAGE_OP_WORDS,
    image_ops_in,
    is_ai_video,
    explicit_media_request,
    TOOL_ABILITY,
    able,
    setup,
    system_prompt,
    warm,
    MEDIA_DONE,
    LIST_ASK,
    OTHER_WORK,
)
from .media import (  # noqa: F401
    explicit_speech_output,
    words_to_speak,
    speech_output_path,
    _clean_spoken_line,
    explicit_transcription_request,
    transcription_paths,
    _unique_workspace_path,
    explicit_media_plan,
    video_wants_narration,
    extract_requested_narration,
    VIDEO_IMAGE_EXTS,
    _selected_video_images,
    _video_call_sources,
)
from .scope import (  # noqa: F401
    error_words,
    _content,
    _overwrites_own,
    OutOfScope,
    _norm,
    in_scope,
    _header,
    READ_ONLY,
    WRITE_WORDS,
    LIST_WRITE,
    CANT,
    CLAIM,
    LONG_WAIT_S,
    LONG_NOTES,
    ARTIFACT_TOOLS,
    JOB_TOOLS,
    REHEARSED,
    plain_error,
    artifact_action_allowed,
    long_note,
)
from .slides_fill import (  # noqa: F401
    fill_slide_shortfall,
    repair_slide_placeholders,
    preserve_explicit_sheet_row,
)
from .research import (  # noqa: F401
    gather,
    sources_md,
    research,
)


def run(state, messages, app="", path="", emit=print, auto_confirm=None, cache=True):
    """Run one agent job. `messages` is the conversation so far (user turns last).

    auto_confirm: None asks the person; "yes"/"no" answers every question at once
    (the checker and the tests, which run with nobody at the screen).
    cache=False: the engine answers nothing from its reply cache, so the model
    really writes every step. Orientation measures the model, not the cache: 53%
    of a pass-2 run came back in under 2 s on repeated wording (2026-10-07)."""
    sess, ws = state.session, state.ws
    sess.reset_stop()
    tier = state.model_card()["tier"]
    mode = state.mode
    perm = state.permission
    asked = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"), "")
    cue = cues.fix(asked)          # routing reads the typo-corrected words; the model keeps asked
    apps, names, job_system = setup(state, app, cue)

    def exists(p):
        try:
            return ws.path(p).exists()
        except ToolError:
            return False
    # a short follow-up to a file request is that request again (shepherd.follow_on, 2026-10-20)
    if tier >= 2 and perm != "plan":
        merged = shepherd.follow_on(messages, exists)
        if merged and not shepherd.intent(asked, app, path, [], "", exists) and not wants_tools(cue, app):
            asked, cue = merged, cues.fix(merged)
            apps, names, job_system = setup(state, app, cue)
    # Files attached to this message are made readable first (Word/Excel/PowerPoint
    # imported, PDF text, sound and video transcribed) and named in the request, so
    # "summarize this" or "add a slide to this" works on them like any named file.
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
    raw_attached = list(last_user.get("attachments") or [])
    missing_video_attachments = [a for a in raw_attached if Path(str(a)).suffix.lower() in VIDEO_IMAGE_EXTS and not exists(a)]
    attached = [a for a in raw_attached if exists(a)]
    att_lines = []
    if attached:
        from .. import attachments as A
        emit({"type": "thinking", "route": "local", "what": "Reading the attached file" + ("s" if len(attached) > 1 else "")})
        ictx = T.Ctx(ws, "agent")
        ictx.state = state               # a picture is read by the model the person chose
        got = A.intake(ws, attached, ictx)
        att_lines = [f"{p} ({what})" for p, what in got]
        asked = asked + "\n\nAttached: " + "; ".join(att_lines)
        cue = cues.fix(asked)
        apps, names, job_system = setup(state, app, cue)
    # A request that needs a kind of model this computer does not have: say which,
    # and where to add it, instead of a text model guessing (it answered "I'm a
    # text-based AI, I can't view images", 2026-10-13). The text model stays the
    # one that understands requests; the others are its tools.
    knows = hasattr(getattr(state, "orch", None), "abilities")      # no orchestrator: nothing is known missing
    for need_pattern, tool, kind in MEDIA_NEEDS if knows else []:
        if not re.search(need_pattern, cue, re.I):
            continue
        # the answer is a checklist: what this kind of work needs, what is here, what to add
        try:
            from ..server import setup_checklist
            items = setup_checklist(state, MEDIA_KIND[tool])
        except Exception:
            items = []
        # GPU support only makes the work faster: it never blocks a request (2026-10-27: an
        # unready AMD runtime stopped picture requests whose model was ready)
        missing = [i for i in items if not i["ok"] and not i.get("speedup")]
        if (tool not in names and not able(state, [tool])[0]) or missing:
            show = [i for i in items if not i["ok"]] or [{"label": f"a {kind} model", "ok": False, "need":
                    f"Sheep needs a {kind} model for this.", "action": {"do": "slot", "slot": "chat", "label": "Add a model"}}]
            names_ = ", ".join(i["label"] for i in show if not i.get("speedup"))
            emit({"type": "text", "route": "local", "text":
                  f"Sheep is not set up for this yet: for {kind} work it needs {names_ or 'one more piece'}. "
                  "Add it with the button below, then press Try again."})
            emit({"type": "setup_needed", "kind": kind, "items": show})
            emit({"type": "done", "stopped": False, "changed": []})
            return
    selected_video_images = _selected_video_images(ws, attached, path, asked)
    # the chat's last reply and the files it made: "add that to the ppt you made
    # before" works on them (the reply carries its files; names said in it count)
    prev_answer = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "assistant"), "")
    previous_request = next((str(m.get("content") or "") for m in reversed(messages[:-1])
                             if m.get("role") == "user"), "")
    last_files = [f for m in messages if m.get("role") == "assistant" for f in (m.get("files") or [])]
    last_files += re.findall(r"[\w./-]+\.(?:sdeck|ssheet|sdoc)\b", prev_answer)
    # making or filling a deck, document or sheet: the model writes, Sheep operates
    job = (shepherd.intent(asked, app, path, last_files, prev_answer, exists)
           if tier >= 2 and perm != "plan" and not explicit_media_request(cue) else None)
    if (not job and cue != asked and tier >= 2 and perm != "plan" and not explicit_media_request(cue)):
        # "mak a powerpont about water": the job is read from the corrected words (file names
        # and capitalised names are never corrected, cues.py)
        job = shepherd.intent(cue, app, path, last_files, prev_answer, exists)
    if job and job.get("reuse") and prev_answer:
        job = dict(job, source_answer=prev_answer, source_request=previous_request)
    if not app and ability_question(cue):
        job, names = None, []        # "can you make spreadsheets?" -- answered, not done
    if not wants_tools(cue, app) and not job:
        names = []                   # a question: answered in words, no tools on offer
    local, cloud = state.local, state.cloud
    cloud_on = mode == 1 and cloud.ready()

    ctx_lines = []
    plain = not names and not app          # a question in the chat: no files, no tools in its prompt
    if path and not plain:
        ctx_lines.append(f"The file open in the editor is: {path}")
    if selected_video_images and ("make_video" in names or re.search(PHOTO_MONTAGE, asked, re.I)):
        ctx_lines.append("Photos available for this video request (attached or selected in the open Studio project): "
                         + ", ".join(selected_video_images))
    # files the request names: the job's target. Their outline rides with the request, so a
    # small model knows slide and block numbers without a read step, and is not tempted by
    # the other files in the workspace listing (2026-10-06 orientation)
    named = [n for n in dict.fromkeys(re.findall(r"[\w./-]+\.(?:sdeck|sdoc|ssheet|sstudio|txt|md|csv|py)\b", asked))
             if exists(n)]
    if named and not plain:
        reader = {".sdeck": "slides_read", ".sdoc": "words_read", ".ssheet": "sheets_read"}
        for n in named[:2]:
            r = reader.get("." + n.rsplit(".", 1)[-1].lower())
            if r:
                try:
                    ctx_lines.append(f"{n} holds now:\n" + str(T.run(ws, r, {"path": n}, by="agent"))[:1200])
                except ToolError:
                    pass
    try:
        files = [f["path"] for f in ws.tree()][:25] if not plain else []
        if not plain and not named:
            ctx_lines.append("Workspace files: " + (", ".join(files) if files else "(empty)"))
        elif not plain:
            ctx_lines.append("Work on: " + ", ".join(named))
    except ToolError:
        pass
    # The system turn and the tool list are the same on every run of an app, so
    # the engine's slot cache reads them once. What changes run to run -- the
    # open file, the workspace listing -- rides at the END of the user's turn;
    # placed in the system turn it broke that reuse and every first step paid
    # for ~2.5K tokens again (116 s on the first live run).
    # A plain question gets a plain system turn: shown the tool-and-workspace
    # one, the 3B answered "How tall is Mount Elgon?" by writing herder code
    # about notes.sdoc as text (2026-10-05).
    # The job turn is the one warm() read into the engine (setup), so its cache holds.
    system = answer_prompt(state) if plain else job_system
    convo = [{"role": "system", "content": system}] + [{k: v for k, v in m.items() if k not in ("files", "attachments")}
                                                       for m in messages if m.get("role") != "system"]
    if att_lines:
        ctx_lines.insert(0, "Attached to this message: " + "; ".join(att_lines))
    last_user = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"), "")
    for m in reversed(convo):
        if m.get("role") == "user" and ctx_lines:
            m["content"] = f"{m.get('content') or ''}\n\n({'; '.join(ctx_lines)})"
            break

    # executor: the local model unless it has not earned tools and the cloud is on
    executor = local
    stand_in = False          # the cloud answers only because the local reply was gibberish
    orch = getattr(state, "orch", None)

    def base_kw(**extra):
        """What every local call passes: the doctor's frame and sampling, and the
        reply cache off when this job asked for it."""
        kw = dict(orch.chat_kwargs() if orch else {})
        if not cache:
            kw["cache"] = False
        if executor is local and getattr(local, "retry_device", None):
            kw["on_retry"] = lambda why: emit({"type": "device_retry", "why": why})
        kw.update(extra)
        return kw

    if orch and orch.holding():
        # the Egypt protocol: a model that produced gibberish is being fixed;
        # its replies are held back until the canaries pass again
        if cloud_on:
            executor = cloud
            stand_in = True
            names = able(state, T.offered(1 if perm == "plan" else 3, apps))[0]
            emit({"type": "route", "route": "cloud", "why": "your model is being fixed; Nebius answers meanwhile"})
        else:
            emit({"type": "text", "route": "local", "text": "Your model produced faulty replies, so the orchestrator is "
                  "fixing it (Settings › Activity shows each step). Its replies are held back until it passes its checks again."})
            emit({"type": "done", "stopped": False, "changed": []})
            return
    elif orch and not orch.ready():
        # Plan P (Ernest): while the orchestrator gets the local model ready,
        # Nemotron answers in Sheep + Nebius mode; in Sheep mode the turn waits,
        # with the stage and a progress figure for the loading bar
        if cloud_on:
            executor = cloud
            names = able(state, T.offered(1 if perm == "plan" else 3, apps))[0]
            emit({"type": "route", "route": "cloud", "why": "Nemotron on Nebius answers" if not getattr(state, "model_path", "")
                  else "your model is getting ready; Nemotron answers meanwhile"})
        else:
            t0, last = time.time(), None
            while not orch.ready() and not orch.holding() and time.time() - t0 < 900:
                if sess.stop.is_set():
                    emit({"type": "done", "stopped": True, "changed": []})
                    return
                now = (orch.stage, round(orch.progress, 2))
                if now != last:
                    emit({"type": "waiting", "stage": orch.stage, "progress": orch.progress})
                    last = now
                time.sleep(0.5)
            if not orch.ready():
                emit({"type": "text", "route": "local", "text": "Sheep could not get your model ready. Settings › Activity "
                      "shows what went wrong."})
                emit({"type": "done", "stopped": False, "changed": []})
                return
            emit({"type": "waiting", "stage": "ready", "progress": 1.0})
    # A question about now (not a job on files): the orchestrator researches it
    # itself, as it does exact arithmetic. Given web tools, the 3B searched once,
    # then wandered into files and herders and never answered (2026-10-05).
    from ..tools import web as _web
    if _web.STATE["on"] and re.search(WEB_CUES, cue.lower()) and not re.search(ACTION, cue.lower()) and not job:
        return research(asked, convo, executor, emit)
    if tier == 0 and cloud_on:
        executor = cloud
        names = able(state, T.offered(1 if perm == "plan" else 3, apps))[0]
        emit({"type": "route", "route": "cloud", "why": "the local model has not earned tools (tier 0)"})
    elif tier == 0:
        names = []

    # A deck or document about something current: search first, and whoever writes it works from
    # what was found; the file ends with its sources (hackathon B7: Tavily in a real job)
    if job and job["kind"] in ("slides", "words") and _web.STATE["on"] and re.search(WEB_CUES, cue.lower()):
        found_res, found = gather(asked, emit)
        if found_res:
            job = dict(job, research=found, sources=found_res[:5])
            ctx_lines.append("Current facts found on the web; use them, never invent figures, and end with a "
                             "Sources slide or section listing these links:\n" + found[:2500])
            emit({"type": "sources", "items": found_res[:5]})

    # Mode 1: Nemotron writes the job as exact tool calls, content included, and
    # SHEEP runs them here at once. Planned in words, the 3B re-typed a deck slide
    # by slide at about a minute a step while the chat looked stuck (2026-10-01).
    exec_calls = []
    # adding to a file that exists is the local model's to write and Nemotron's to check, as the video shows;
    # planned by Nemotron, there was no draft to check (journey J5, 2026-10-22)
    local_extends = bool(job and job.get("action") == "extend" and job.get("kind") in ("slides", "words", "sheets"))
    if (cloud_on and executor is local and perm != "plan" and names and not local_extends
            and not explicit_transcription_request(cue)):
        emit({"type": "thinking", "route": "cloud", "what": "Planning with Nemotron"})
        # the web was already read for this job (its facts are in the context): a plan that searches again
        # handed the deck to the local loop, which could not finish it (journey J4, 2026-10-22)
        plan_names = [n for n in names if not n.startswith("web_")] if job and job.get("research") else names
        try:
            spec = json.dumps(T.schemas(plan_names), ensure_ascii=False)
            plan_msg, info = cloud.chat([{"role": "system", "content": PLANNER_CALLS},
                                         {"role": "user", "content": f"Tools (JSON schemas):\n{spec}\n\n"
                                          f"{chr(10).join(ctx_lines)}\n\nJob: {last_user}"}],
                                        max_tokens=8000, why="plan the job as tool calls")  # it reasons first
            exec_calls = _plan_calls(plan_msg.get("content") or "", set(plan_names))
            if exec_calls and any(T.TOOLS[c["tool"]].tier < 2 for c in exec_calls):
                # the job needs what is in a file: the local loop reads, then acts
                plan = "\n".join(f"{i + 1}. {c['tool']} {json.dumps(c['args'], ensure_ascii=False)[:200]}"
                                 for i, c in enumerate(exec_calls))
                convo.insert(1, {"role": "system", "content": "Follow this plan, one tool call at a time:\n" + plan})
                emit({"type": "plan", "steps": len(exec_calls), "text": plan, "route": "cloud", "ms": info["ms"]})
                exec_calls = []
            elif exec_calls:
                emit({"type": "plan", "steps": len(exec_calls), "route": "cloud", "ms": info["ms"],
                      "text": "\n".join(f"{i + 1}. {c['tool']} {json.dumps(c['args'], ensure_ascii=False)[:200]}"
                                        for i, c in enumerate(exec_calls))})
        except LLMError as e:
            emit({"type": "route", "route": "local", "why": f"Nemotron unavailable ({e}); Sheep works alone"})

    schemas = T.schemas(names)
    rc = T.Ctx(ws, "agent")          # one context for the whole job: notes, exports, files made
    rc.state = state
    rc.cancel_event = sess.stop
    rc.register_cancel = sess.set_cancel_hook
    fails = 0
    tool_oks, tool_fails = [], []     # this job's tool results, for the honesty check before "done"

    def ask(cid, name, args, reason):
        """The person's yes or no; the tests and the checker answer for themselves."""
        if auto_confirm in ("yes", "no"):
            emit({"type": "confirm", "id": cid, "name": name, "args": args, "reason": reason,
                  "auto": auto_confirm})
            return auto_confirm == "yes"
        ev = threading.Event()
        sess.pending[cid] = {"event": ev, "allow": False}
        emit({"type": "confirm", "id": cid, "name": name, "args": args, "reason": reason})
        ev.wait(timeout=600)
        return sess.pending.pop(cid, {}).get("allow", False)

    scope_asked = asked

    def call(cid, name, args, route):
        """Run one tool for the job; returns (ok, text) and emits what happened."""
        if getattr(state, "rehearsal", False) and name in REHEARSED:
            text = f"(app check) {name} would run now"
            emit({"type": "tool_result", "id": cid, "name": name, "ok": True, "text": text, "ms": 0, "route": route})
            acted.add(name)
            return True, text
        before = set(rc.made)
        out_key = {"speak_text": "path", "make_image": "path", "make_video": "path", "edit_image": "out"}.get(name)
        if out_key and args.get(out_key) and exists(args[out_key]) and args[out_key] not in rc.made:
            # a new recording or picture never takes an existing file's name: the 3B named its voice
            # file meeting.wav, was refused, and said it was saved (2026-10-14 live run)
            args = {k: v for k, v in args.items() if k != out_key}
        try:
            if not artifact_action_allowed(name, scope_asked) and not (job and name in JOB_TOOLS.get(job.get("kind"), ())):
                raise ToolError("Nothing was changed: this turn did not ask to create or render an artifact. "
                                "Ask explicitly if you want a new file or video.")
            # the job's own target counts as asked for: "add three slides to the deck" names no file, and
            # the deck Sheep resolved from the chat was refused as "not part of this request" (B10, 2026-10-21)
            target = f"\nWork on: {job['target']}" if job and job.get("target") else ""
            why = in_scope(ws, name, args, scope_asked + target, path, rc.made) or _overwrites_own(ws, name, args, rc.made)
            if why:
                raise OutOfScope(why)
            reason = T.needs_confirm(ws, name, args, rc)
            if perm == "ask" and T.TOOLS[name].tier >= 2:
                reason = reason or f"{name} {json.dumps(args)[:200]}"
            if reason and not ask(cid, name, args, reason):
                raise ToolError("the user declined this action; do not retry it, ask what they want instead")
            if not sess.begin_action(name):
                raise ToolError("the person stopped this turn before the action started")
            t0 = time.perf_counter()
            # a step that runs long says what it is doing (a video takes minutes)
            def report_long_step():
                emit({"type": "thinking", "route": "local", "tool_id": cid, **long_note(name, args)})
            slow = threading.Timer(LONG_WAIT_S, report_long_step)
            slow.daemon = True
            slow.start()
            action_open = True
            try:
                try:
                    if sess.stop.is_set():
                        raise ToolError("the person stopped this turn before the action started")
                    rc.progress = lambda update: emit({"type": "tool_progress", "id": cid, "name": name,
                                                       **(update if isinstance(update, dict) else {})})
                    result = T.run(ws, name, dict(args), by="agent", ctx=rc)
                except ModelSelectionRequired as required:
                    # Pause before any download. The UI opens the typed local model
                    # chooser; the same tool call resumes only after Check and use.
                    sess.end_action(name)
                    action_open = False
                    if not wait_for_model_selection(sess, cid, required, emit):
                        raise ToolError(f"{required}. The request was not resumed.")
                    if not sess.begin_action(name):
                        raise ToolError("the person stopped this turn before the feature model was ready")
                    action_open = True
                    result = T.run(ws, name, dict(args), by="agent", ctx=rc)
            finally:
                rc.progress = None
                slow.cancel()
                sess.set_cancel_hook(None)
                if action_open:
                    sess.end_action(name)
            emit({"type": "tool_result", "id": cid, "name": name, "ok": True, "text": result,
                  "ms": round((time.perf_counter() - t0) * 1000), "route": route})
            ok = True
            if T.TOOLS.get(name) and T.TOOLS[name].tier >= 2:
                # a change that names no made file (copy, move, delete, calendar, mail)
                # still counts: the no-tool guard said "nothing was changed" after a copy
                acted.add(name)
        except ToolError as e:
            result = str(e)
            ok = False
            emit({"type": "tool_result", "id": cid, "name": name, "ok": False, "text": result, "route": route})
        except Exception as e:     # a tool bug must not kill the run; the model hears about it
            result = f"{type(e).__name__}: {e}"
            ok = False
            emit({"type": "tool_result", "id": cid, "name": name, "ok": False, "text": result, "route": route})
        for f in sorted(rc.made - before):
            emit({"type": "file_changed", "path": f})
        return ok, result

    def summary():
        """What the job did, in words, for when the orchestrator ends it."""
        if not rc.made:
            return "Done. Nothing needed changing."
        return "Done: " + ", ".join(sorted(rc.made)) + (" updated." if len(rc.made) > 1 else " is ready.")

    seen_calls = set()
    added = set()
    called_tools = set()
    acted = set()             # tools that changed something this job (tier 2+), made file or not
    gave_up = False           # the job ended without doing what was asked (shown as failed)
    checked = claimed = rechecked = cant_told = False

    def ensure_explicit_speech_output():
        """Complete an explicitly requested voice file if planned tools omitted it.

        Returns True when this helper handled the request (including a reported
        failure), so callers can finish without presenting the model's claim as
        a completed recording.
        """
        nonlocal scope_asked
        if (not explicit_speech_output(cue)            # a speak_text that FAILED is completed here too
                or any(Path(p).suffix.lower() == ".wav" for p in rc.made)):
            return False
        if "speak_text" not in names:
            emit({"type": "text", "route": "local", "text":
                  "I could not create the recording: Sheep’s Windows voice tool is not available in this setup."})
            return True
        try:
            voice_path = speech_output_path(ws, asked)
            scope_asked = asked + "\nRequested audio output: " + voice_path
        except ToolError as e:
            emit({"type": "text", "route": "local", "text": f"I could not choose a safe audio output path: {e}"})
            return True

        # Use explicitly quoted text when supplied; otherwise ask the local
        # model for one grounded sentence. User photos are never sent to TTS.
        spoken = words_to_speak(asked)
        if not spoken:
            prior = "\n".join(str(m.get("content") or "") for m in messages
                               if m.get("role") == "user")[-1800:]
            try:
                emit({"type": "thinking", "route": "local", "what": "Writing a short voice line"})
                line_msg, _ = local.chat(
                    [{"role": "system", "content":
                      "Write exactly one short, natural sentence for a Windows voice to read. "
                      "Use only subjects and details the user described in their messages. "
                      "Do not claim to see an image. Return only the sentence."},
                     {"role": "user", "content": prior}], None, max_tokens=48,
                    **base_kw(cache=False))
                spoken = _clean_spoken_line(line_msg.get("content"))
            except Exception:
                spoken = _clean_spoken_line("")

        cid = "voice_" + uuid.uuid4().hex[:8]
        args = {"text": spoken, "path": voice_path}
        emit({"type": "tool_call", "id": cid, "name": "speak_text", "args": args, "route": "local"})
        called_tools.add("speak_text")
        ok, result = call(cid, "speak_text", args, "local")
        actual = sorted(p for p in rc.made if Path(p).suffix.lower() == ".wav")
        if ok and actual:
            emit({"type": "text", "route": "local", "text": "Created the spoken audio: " + ", ".join(actual) + "."})
        elif ok:
            emit({"type": "text", "route": "local", "text":
                  "The voice tool returned, but Sheep could not verify a saved recording."})
        else:
            emit({"type": "text", "route": "local", "text": f"I could not create the recording: {result}"})
        return True

    def finish(stopped=False, failed=False):
        # The Microsoft files the job asked for are exported by the orchestrator,
        # not left to the model's memory: a deterministic last step, like the
        # exact arithmetic, never forgets.
        stopped = stopped or sess.stop.is_set()
        if not stopped:
            for native, office in list(rc.exports.items()):
                if ws.path(native).exists():
                    cid = "call_export_" + uuid.uuid4().hex[:6]
                    args = {"path": native, "to": office}
                    emit({"type": "tool_call", "id": cid, "name": "export_file", "args": args, "route": "sheep",
                          "auto": True})
                    call(cid, "export_file", args, "sheep")
        if getattr(rc, "sources", None):
            emit({"type": "sources", "items": rc.sources[:8]})
        emit({"type": "done", "stopped": stopped, "failed": bool(failed) and not stopped, "changed": sorted(rc.made)})

    def write(job, outline, again=False, vary=False, notes=""):
        """The model writes the content (markdown), streamed into the chat as it goes.
        vary: a second round for more of the same kind; greedy, the 3B wrote the slide
        it had just written again, word for word (2026-10-06), so it samples instead."""
        noun = shepherd.KINDS[job["kind"]]["noun"]
        emit({"type": "thinking", "route": "local", "what": f"Writing the {noun}"})
        system = shepherd.writer_prompt(job, outline) + ("\nUse exactly that format, starting now." if again else "")
        if job.get("research"):
            system += "\nUse these sources for current facts, and never invent figures:\n" + job["research"][:3500]
        if notes:
            system += "\nA reviewer read your first draft and asked for these fixes; write the whole thing again " \
                      "with them made:\n- " + "\n- ".join(notes)
        kw = base_kw()
        kw["on_delta"] = lambda piece: emit({"type": "draft", "delta": piece})
        if vary:
            kw.update(temperature=0.7, top_p=0.9, cache=False)
        first = (shepherd.deck_slides(ws, job["target"]) or [("", False)])[0][0] if job["action"] == "extend" else ""
        msg, _ = executor.chat([{"role": "system", "content": system},
                                {"role": "user", "content": shepherd.user_turn(job, asked, first.title())}], None,
                               max_tokens=2048, **kw)
        out = msg.get("content") or ""
        trace = os.environ.get("SML_SHEPHERD_TRACE")      # a file: every writing round, asked and written
        if trace:
            with open(trace, "a", encoding="utf-8") as f:
                f.write(f"=== {time.strftime('%H:%M:%S')} {asked!r} vary={vary} again={again}\n--- system\n{system}\n"
                        f"--- written\n{out}\n\n")
        return out

    def add_requested_slide_images(job, parsed, made):
        pic_note = ""
        if job["action"] != "create" or not re.search(DECK_PICTURES, cue, re.I):
            return pic_note, False
        deck = re.sub(r"\.(pptx)$", ".sdeck", made)
        slides = [s for s in (parsed[2] if len(parsed) > 2 else []) if s][:3]
        if not able(state, ["make_image"])[0]:
            pic_note = "\n\nNo pictures yet: this computer has no picture model (Settings › Models › pictures)."
        for n, slide in enumerate(slides):
            if sess.stop.is_set():
                return pic_note, True
            if pic_note:
                break
            title = (slide.get("title") if isinstance(slide, dict) else str(slide)) or "slide"
            topic = (parsed[0] if parsed and isinstance(parsed[0], str) else "") or ""
            prompt = f"{title}, {topic}, clean professional illustration, no text".strip(", ")
            img = f"media/{shepherd.slug(title, '')[:40]}-{n + 1}.png"
            cid = f"shep_img_{n}_" + uuid.uuid4().hex[:4]
            args = {"prompt": prompt, "path": img}
            emit({"type": "tool_call", "id": cid, "name": "make_image", "args": args, "route": "sheep"})
            ok, result = call(cid, "make_image", dict(args), "sheep")
            if not ok:
                pic_note = f"\n\nThe pictures were not made: {result.split('. ')[0]}."
                break
            cid = f"shep_place_{n}_" + uuid.uuid4().hex[:4]
            args = {"path": deck, "slide": n + 2, "image": img, "x": 56, "y": 28, "width": 38}
            emit({"type": "tool_call", "id": cid, "name": "slides_add_image", "args": args, "route": "sheep"})
            call(cid, "slides_add_image", dict(args), "sheep")
        return pic_note, False

    def shepherd_job(job, written=""):
        """Make or fill the file from what the model wrote (or already wrote in its
        last reply), with the app's tools, then check it against the request."""
        kind = job["kind"]
        outline = ""
        if (kind == "sheets" and job["action"] == "extend" and re.search(shepherd.TOTAL_ROW, asked.lower())
                and not re.search(r"\brows? (for|of|with|about)\b", asked.lower())):
            # "add a total row": Sheep writes the SUM formulas itself, exactly
            tr = shepherd.total_row(ws, job["target"])
            if not tr:
                emit({"type": "text", "route": "local", "text": f"{job['target']} has no numbers to total, or a Total "
                      "row already."})
                return finish()
            cid = "shep_total_" + uuid.uuid4().hex[:4]
            args = {"path": job["target"], "cell": tr[0], "values": [tr[1]]}
            emit({"type": "tool_call", "id": cid, "name": "sheets_write", "args": args, "route": "sheep"})
            ok, result = call(cid, "sheets_write", args, "sheep")
            emit({"type": "text", "route": "local", "text": (f"Added a Total row to {job['target']} (row "
                  f"{tr[0][1:]}, summing {', '.join(c for c in tr[1] if c.startswith('='))})." if ok else result)})
            return finish()
        have = shepherd.deck_titles(ws, job["target"]) if kind == "slides" and job["action"] == "extend" else []
        deck_title = have[0] if have else ""
        all_slides = shepherd.deck_slides(ws, job["target"]) if kind == "slides" and job["action"] == "extend" else []
        if kind == "slides" and job.get("says") and len(all_slides) > job["says"]:
            # "only has one slide" when the deck holds more: say what is there, change nothing
            emit({"type": "text", "route": "local", "text": f"{job['target']} has {len(all_slides)} slides: "
                  + ", ".join(t.title() for t, _ in all_slides[:12] if t) + ". Tell me what to add, or open another "
                  "deck."})
            return finish()
        if job["action"] == "extend":
            reader = {"slides": "slides_read", "words": "words_read", "sheets": "sheets_read", "text": "read_file",
                      "code": "read_file"}[kind]
            try:
                outline = T.run(ws, reader, {"path": job["target"]}, by="agent", ctx=rc)
            except ToolError:
                outline = ""
            if kind in ("text", "code"):
                try:
                    job = dict(job, existing=ws.path(job["target"]).read_text(encoding="utf-8"))
                except (OSError, ToolError, UnicodeDecodeError):
                    job = dict(job, existing="")
        if have:
            job = dict(job, taken=[t.title() for t in have])
        cols = []
        if kind == "sheets" and job["action"] == "extend":
            # the sheet's own header: the writer is told it, and a written table without it
            # is all new rows (the row asked for was taken as a header and lost; 2026-10-06)
            grid = shepherd.last_row(ws, job["target"]) and _header(ws, job["target"])
            cols = grid or []
            job = dict(job, columns=cols)
        text = written or write(job, outline)
        if job.get("sources") and kind in ("slides", "words") and "## Sources" not in text:
            text += sources_md(job["sources"])
        parsed = shepherd.parse(kind, text)
        if written and job.get("reuse") and shepherd.reuse_needs_reformat(kind, written, parsed):
            text = write(job, outline, again=True)
            parsed = shepherd.parse(kind, text)
            written = ""  # permit the bounded retry if the reformatted draft is malformed
        if cols and parsed[0] and [str(h).strip().lower() for h in parsed[0]] != [str(c).strip().lower() for c in cols]:
            parsed = ([], [[shepherd._num(str(c)) for c in parsed[0]]] + parsed[1])
        if have:
            parsed = shepherd.drop_repeats(parsed, have, deck_title)
        if not shepherd.enough(kind, parsed) and not written:
            text = write(job, outline, again=True)
            parsed = shepherd.parse(kind, text)
            if have:
                parsed = shepherd.drop_repeats(parsed, have, deck_title)
        if kind == "slides":
            parsed = shepherd.drop_thin(shepherd.merge_repeats(parsed))
            # The model can provide a partial draft and still need another writing
            # round. Do not gate this on `written`: that caused 1/2-slide decks.
            parsed = fill_slide_shortfall(job, parsed, have, deck_title, outline, write)
            parsed = repair_slide_placeholders(job, parsed, have, deck_title, outline, write)
        elif kind == "sheets" and job["action"] == "extend":
            parsed = preserve_explicit_sheet_row(asked, cols, parsed)
        if have and not shepherd.enough(kind, parsed):
            emit({"type": "text", "route": "local", "text": f"Nothing new to add: {job['target']} already has those "
                  "slides. Tell me which topics to add."})
            return finish()
        if (cloud_on and executor is local and kind in ("slides", "words", "sheets") and not written
                and shepherd.enough(kind, parsed)):
            # Sheep + Nebius: Nemotron checks the local model's draft against the request before Sheep
            # builds the file, and the local model rewrites it once with the fixes (hackathon B6)
            emit({"type": "thinking", "route": "cloud", "what": "Nemotron is checking the draft"})
            try:
                msg, _ = cloud.chat([{"role": "system", "content": REVIEW.format(noun=shepherd.KINDS[kind]["noun"])},
                                     {"role": "user", "content": f"Request: {asked}\n\nDraft:\n{text[:6000]}"}],
                                    max_tokens=3000, why="check the local model's draft")
                verdict = review_verdict(msg.get("content") or "")
            except LLMError:
                verdict = None
            if verdict is not None:
                fixed = False
                if verdict["fixes"]:
                    again_text = write(job, outline, again=True, notes=verdict["fixes"])
                    again_parsed = shepherd.parse(kind, again_text)
                    if shepherd.enough(kind, again_parsed):
                        text, parsed, fixed = again_text, again_parsed, True
                        if kind == "slides":
                            parsed = shepherd.drop_thin(shepherd.merge_repeats(parsed))
                emit({"type": "review", "by": cloud.model, "ok": verdict["ok"], "fixes": verdict["fixes"],
                      "fixed": fixed})
        if not shepherd.enough(kind, parsed):
            emit({"type": "text", "route": "local", "text": "I could not write this in a form Sheep can put into a "
                  f"{shepherd.KINDS[kind]['noun']}. Try saying what it should contain."})
            return finish()
        todo = shepherd.calls(job, parsed, exists, lambda p: shepherd.last_row(ws, p) + 1)
        for i, (tool, args) in enumerate(todo):
            if sess.stop.is_set():
                return finish(stopped=True)
            cid = f"shep_{i}_" + uuid.uuid4().hex[:4]
            emit({"type": "tool_call", "id": cid, "name": tool, "args": args, "route": "sheep"})
            ok, result = call(cid, tool, dict(args), "sheep")
            if i == 0 and job["action"] == "create" and "already exists" in result:
                emit({"type": "text", "route": "local", "text": result.split(";")[0] + ". Say \"add to it\" to fill "
                      "it, or \"replace it\" to start it again."})
                return finish()
            if sess.stop.is_set():
                return finish(stopped=True)
            if not ok:
                emit({"type": "text", "route": "local", "text": f"I couldn't finish this: {plain_error(result)}"})
                return finish(failed=True)
        made = todo[0][1]["path"] if job["action"] == "create" else job["target"]
        pic_note, stopped = add_requested_slide_images(job, parsed, made) if kind == "slides" else ("", False)
        if stopped:
            return finish(stopped=True)
        if job.get("pdf") and kind in ("slides", "words", "sheets"):
            # "... as a PDF": Sheep exports it, and checks the PDF it wrote
            native_now = re.sub(r"\.(pptx|docx|xlsx)$", shepherd.KINDS[kind]["ext"], made)
            cid = "shep_pdf_" + uuid.uuid4().hex[:4]
            args = {"path": native_now, "format": "pdf"}
            emit({"type": "tool_call", "id": cid, "name": "export_file", "args": args, "route": "sheep"})
            call(cid, "export_file", args, "sheep")
        native = re.sub(r"\.(pptx|docx|xlsx)$", shepherd.KINDS[kind]["ext"], made)
        short = verify.shortfalls(ws, asked, rc.made)
        if kind == "slides" and job["action"] == "extend" and job["count"] and len(parsed[2]) < job["count"]:
            short.append(f"{len(parsed[2])} of the {job['count']} slides asked were written")
        note = ("\n\nNot yet as asked: " + "; ".join(short) + ".") if short else ""
        emit({"type": "text", "route": "local", "text": shepherd.summary(job, parsed, native) + note + pic_note})
        return finish()

    # "What files do I have?" Sheep reads the folder; the model answers from that
    # listing. If its answer names none of the real files, the listing is shown as
    # it stands -- the question has a plain answer and nobody should get "I have no
    # access to your files" (2026-10-07 orientation).
    if not named and not job and re.search(LIST_ASK, cue.lower()) and not re.search(LIST_WRITE, cue.lower()):
        emit({"type": "thinking", "route": "local", "what": "Reading the workspace"})
        try:
            listing = str(T.run(ws, "list_files", {}, by="agent"))
        except ToolError as e:
            listing = f"(the workspace could not be read: {e})"
        names = [p.name for p in ws.root.iterdir()] if ws.root.exists() else []
        said_text = ""
        if executor is local and names:
            msgs = [{"role": "system", "content": ANSWER + " These are the files in the person's workspace. "
                     "Tell them what is there, naming the files exactly as written. Do not say you cannot "
                     "see their files: the list below is theirs."},
                    {"role": "user", "content": asked + "\n\nThe workspace holds:\n" + listing[:6000]}]
            try:
                kw = base_kw()
                kw["on_delta"] = lambda piece: emit({"type": "draft", "delta": piece})
                msg, info = executor.chat(msgs, None, **kw)
                orch.after_turn(info) if orch else None
                said_text = (msg.get("content") or "").strip()
            except LLMError:
                said_text = ""
        if not any(n.lower() in said_text.lower() for n in names):
            said_text = ("Your workspace is empty." if not names
                         else "Here is what your workspace holds:\n\n" + listing)
        emit({"type": "text", "route": "local", "text": said_text})
        return finish()

    # A question about a named file ("summarize report.sdoc", "what is the total in budget.ssheet?",
    # "read notes.txt and tell me what it says") is answered from the file, with no tools: asked to
    # summarize, the 3B wrote its summary into the person's document (2026-10-06 orientation).
    if named and not job and re.search(READ_ONLY, cue.lower()) and not re.search(WRITE_WORDS, cue.lower()):
        reader = {".sdeck": "slides_read", ".sdoc": "words_read", ".ssheet": "sheets_read"}
        parts = []
        for n in named[:3]:
            try:
                parts.append(f"--- {n} ---\n" + str(T.run(ws, reader.get("." + n.rsplit(".", 1)[-1].lower(), "read_file"),
                                                         {"path": n}, by="agent"))[:6000])
            except ToolError as e:
                parts.append(f"--- {n} ---\n(could not be read: {e})")
        emit({"type": "thinking", "route": "local", "what": "Reading " + ", ".join(named[:3])})
        msgs = [{"role": "system", "content": ANSWER + " Answer from the file contents given below; quote names and "
                 "numbers exactly as they are written there."},
                {"role": "user", "content": asked + "\n\n" + "\n\n".join(parts)}]
        try:
            if executor is local:
                kw = base_kw()
                kw["on_delta"] = lambda piece: emit({"type": "draft", "delta": piece})
                msg, info = executor.chat(msgs, None, **kw)
            else:
                msg, info = executor.chat(msgs, None, why="answer a question about a file")
        except LLMError as e:
            emit({"type": "error", "text": error_words(e, local=executor is local)})
            return finish()
        emit({"type": "text", "text": (msg.get("content") or "").strip(), "route": info.get("route", "local")})
        return finish()

    # A Studio video to make: planned from the request (template, length, shape, title), made
    # with one call, rendered when asked. AI video ("a video of a dog running") is make_video's.
    sp = (shepherd.studio_plan(asked) if tier >= 2 and perm != "plan" and not job and app in ("", "studio")
          and not is_ai_video(cue) and not re.search(PHOTO_MONTAGE, cue, re.I) else None)
    if sp and not exec_calls and executor is local:
        if not sp["title"]:
            emit({"type": "thinking", "route": "local", "what": "Writing the title"})
            kw = base_kw()
            msg, _ = executor.chat([{"role": "system", "content": "Give a title for this video in 2 to 6 words on line 1 "
                                     "and a one-line subtitle on line 2. Nothing else."},
                                    {"role": "user", "content": asked}], None, max_tokens=40, **kw)
            lines = [l.strip(" #*\"'") for l in (msg.get("content") or "").splitlines() if l.strip()]
            sp["title"] = (lines[0] if lines else "Welcome")[:60]
            sp["subtitle"] = sp["subtitle"] or (lines[1][:60] if len(lines) > 1 else "")
        # the name the person gave, else one from the title; never over an existing project
        base = sp.get("path") or shepherd.slug(sp["title"], ".sstudio")
        pth, k = base, 2
        while exists(pth):
            pth = base[:-len(".sstudio")] + f"-{k}.sstudio"
            k += 1
        todo = [("studio_create", {"path": pth, "template": sp["template"], "duration": sp["duration"],
                                   "format": sp["format"], "title": sp["title"],
                                   **({"subtitle": sp["subtitle"]} if sp["subtitle"] else {})})]
        if sp["render"]:
            todo.append(("studio_render", {"path": pth}))
        for i, (tool, args) in enumerate(todo):
            cid = f"studio_{i}_" + uuid.uuid4().hex[:4]
            emit({"type": "tool_call", "id": cid, "name": tool, "args": args, "route": "sheep"})
            ok, result = call(cid, tool, dict(args), "sheep")
            if sess.stop.is_set():
                return finish(stopped=True)
            if not ok:
                emit({"type": "text", "route": "local", "text": f"I couldn't finish this: {plain_error(result)}"})
                return finish(failed=True)
        kind_word = {"title-intro": "title intro", "logo-reveal": "logo reveal", "outro": "outro",
                     "short": "9:16 short", "lower-third": "lower third"}[sp["template"]]
        emit({"type": "text", "route": "local", "text": f"Made {pth}: a {sp['duration']}-second {kind_word} titled "
              f"“{sp['title']}”" + (", rendered to MP4." if sp["render"] else ". Open it in Studio to change it, or "
                                    "ask me to render it.")})
        return finish()

    # A named slide-title edit is unambiguous. Send it directly to slides_set so
    # a small model cannot rebuild the deck or wander into unrelated tools.
    rename = re.fullmatch(r"\s*rename\s+slide\s+(\d+)\s+(?:of|in)\s+(.+?\.sdeck)\s+to\s+(.+?)\s*", asked, re.I)
    if (rename and not exec_calls and executor is local and tier >= 2 and perm != "plan"
            and "slides_set" in names and exists(rename.group(2).strip())):
        slide = int(rename.group(1))
        target = rename.group(2).strip()
        title = rename.group(3).strip()
        if title.endswith("."):
            title = title[:-1].rstrip()
        cid = "slide_rename_" + uuid.uuid4().hex[:6]
        args = {"path": target, "slide": slide, "title": title}
        emit({"type": "tool_call", "id": cid, "name": "slides_set", "args": args, "route": "local"})
        ok, result = call(cid, "slides_set", args, "local")
        emit({"type": "text", "route": "local", "text": (f"Renamed slide {slide} in {target} to {title}." if ok
                                                                  else f"I could not rename that slide: {result}")})
        return finish()

    # A photo edit on a named picture has one exact action too (2026-10-14 live run: the 3B,
    # offered edit_image, still reached for slides_add_image and make_video five times).
    pic = re.search(r"(?<![\w./\\-])([\w./\\-]+\.(?:png|jpe?g|webp|bmp|gif))\b", asked, re.I)
    edit_ops = image_ops_in(cue) if pic else ""          # typo-corrected: "blak and white"
    if (pic and edit_ops and not exec_calls and executor is local and tier >= 2 and perm != "plan"
            and "edit_image" in names and exists(pic.group(1))
            and not re.search(r"\b(slide|deck|document|doc|report|video|montage)\b", asked, re.I)):
        cid = "photo_edit_" + uuid.uuid4().hex[:6]
        args = {"path": pic.group(1), "ops": edit_ops}
        emit({"type": "tool_call", "id": cid, "name": "edit_image", "args": args, "route": "local"})
        ok, result = call(cid, "edit_image", args, "local")
        emit({"type": "text", "route": "local", "text": (result[0].upper() + result[1:] + "." if ok
                                                          else f"I could not edit that picture: {result}")})
        return finish()

    # A plain folder-creation request has one exact action. Do not let a small
    # model loop on list_files when create_folder is already available.
    folder = re.fullmatch(r"\s*(?:please\s+)?(?:create|make)\s+(?:a\s+)?folder\s+(?:called|named)\s+(.+?)\s*[.!]?\s*",
                          asked, re.I)
    folder_name = folder.group(1).strip().strip("\"'") if folder else ""
    if (folder_name and not re.search(r"\b(?:in|inside|within)\b", folder_name, re.I)
            and not exec_calls and executor is local and tier >= 2 and perm != "plan"
            and "create_folder" in T.TOOLS):
        cid = "folder_create_" + uuid.uuid4().hex[:6]
        args = {"path": folder_name}
        emit({"type": "tool_call", "id": cid, "name": "create_folder", "args": args, "route": "local"})
        ok, result = call(cid, "create_folder", args, "local")
        emit({"type": "text", "route": "local", "text": (f"{result}." if ok
                                                                  else f"I could not create that folder: {result}")})
        return finish()

    if explicit_transcription_request(cue) and not exec_calls and perm != "plan":
        if "transcribe_audio" not in names:
            emit({"type": "text", "route": "local", "text":
                  "I could not transcribe this: the local Whisper tool is not available in this setup."})
            return finish()
        try:
            source, transcript_path = transcription_paths(ws, asked)
            if not source:
                raise ToolError("name the local audio or video file to transcribe")
            ws.path(source, must_exist=True)
        except ToolError as e:
            emit({"type": "text", "route": "local", "text": f"I could not start the transcription: {e}"})
            return finish()
        scope_asked = asked + "\nRequested transcript output: " + transcript_path
        cid = "transcribe_" + uuid.uuid4().hex[:8]
        args = {"path": source, "out": transcript_path}
        emit({"type": "tool_call", "id": cid, "name": "transcribe_audio", "args": args, "route": "local"})
        ok, result = call(cid, "transcribe_audio", args, "local")
        if ok:
            transcript = result.split(":\n", 1)[1].strip() if ":\n" in result else result
            emit({"type": "text", "route": "local", "text":
                  f"Whisper saved the transcript to {transcript_path}:\n\n{transcript}"})
        else:
            emit({"type": "text", "route": "local", "text": f"Whisper could not transcribe the file: {result}"})
        return finish()

    if explicit_media_request(cue) and not exec_calls and executor is local and tier >= 2 and perm != "plan":
        try:
            narration = extract_requested_narration(asked)
            narration_is_explicit = bool(narration)
            wants_narration = video_wants_narration(cue)
            requests_music_effects = bool(re.search(r"\b(music|sound effects?|sfx|soundtrack|background track)\b", asked, re.I))
            if wants_narration and not narration:
                emit({"type": "thinking", "route": "local", "what": "Writing a short voice-over"})
                instruction = ("Write one natural spoken sentence of 4 to 9 words for a video voice-over. "
                               "Return only the sentence. Do not add a preamble, sound effects, or facts not in the request. "
                               "If the request uses selected photos whose contents are not described, use generic wording "
                               "and do not pretend to see details.")
                msg, _ = local.chat([{"role": "system", "content": instruction},
                                     {"role": "user", "content": asked}], None, max_tokens=48,
                                    **base_kw(cache=False))
                narration = re.sub(r"^\s*[\"'`]|[\"'`]\s*$", "", str(msg.get("content") or "").strip())
                words = re.findall(r"\b[\w'-]+\b", narration)
                if not 3 <= len(words) <= 18 or len(narration) > 220 or "```" in narration:
                    narration = ("Your selected photographs unfold in a gentle visual story."
                                 if selected_video_images else "A short scene unfolds in gentle motion.")
            if narration_is_explicit and (len(narration) > 220 or "```" in narration):
                raise ToolError("the requested narration is too long for one voice clip; shorten the spoken line and retry")
            media_calls = explicit_media_plan(ws, asked, selected_video_images, narration)
            if not media_calls:
                raise ToolError("this media request could not be routed safely")
        except (ToolError, LLMError) as e:
            emit({"type": "text", "route": "local", "text": f"I could not prepare this media request: {e}"})
            return finish()

        # All generated paths belong to this explicitly requested media job.
        # Authorize those exact paths through the existing scope guard; arbitrary
        # extra model-suggested outputs remain out of scope.
        planned_paths = []
        for _, args in media_calls:
            planned_paths.extend(str(args[k]) for k in ("path", "out") if args.get(k))
        scope_asked = asked + "\nRequested media outputs: " + "; ".join(planned_paths)
        done = []
        verified_audio = []
        audio_warnings = []
        for index, (name, args) in enumerate(media_calls):
            if sess.stop.is_set():
                return finish(stopped=True)
            if name not in names:
                emit({"type": "text", "route": "local", "text": f"This Sheep setup cannot run {name}; no substitute route was used."})
                return finish()
            cid = f"media_{index}_" + uuid.uuid4().hex[:6]
            emit({"type": "tool_call", "id": cid, "name": name, "args": args, "route": "local"})
            ok, result = call(cid, name, dict(args), "local")
            if not ok:
                partial = "; ".join(sorted(rc.made))
                suffix = f" Valid partial outputs: {partial}." if partial else " No output was completed."
                emit({"type": "text", "route": "local", "text": f"Media creation stopped at {name}: {result}.{suffix}"})
                return finish()
            done.append(str(result))
            if name == "studio_render" and args.get("out") and wants_narration:
                rendered = ws.path(str(args["out"]), must_exist=True)
                if shutil.which("ffprobe"):
                    from ..tools.studio import _probe
                    _, has_audio = _probe(rendered)
                    verified_audio.append(bool(has_audio))
                    if not has_audio:
                        audio_warnings.append(f"ffprobe found no audio track in {args['out']}")
                else:
                    audio_warnings.append(f"could not verify the audio track in {args['out']} because ffprobe is unavailable")
        outputs = ", ".join(sorted(rc.made))
        note = ""
        if wants_narration:
            if verified_audio and all(verified_audio) and not audio_warnings:
                note += " The rendered video audio tracks were verified."
            else:
                detail = "; ".join(audio_warnings)
                note += " Narration was requested, but the video audio track was not verified; do not assume the MP4 contains sound."
                if detail:
                    note += f" {detail}."
        if requests_music_effects:
            note += " This local setup can add spoken narration, but it cannot generate music or sound effects; none were added."
        emit({"type": "text", "route": "local", "text": f"Created the requested media locally: {outputs}.{note}"})
        return finish()

    if not job and not exec_calls and re.match(FILE_ASK, asked, re.I):
        # "i want a file" after a deck was made: the model answered "Here is a simple PowerPoint file
        # created for you" with no file (B10 replay, 2026-10-21). The chat's own files are named instead
        ready = [n for n in dict.fromkeys(Path(str(f)).name for f in last_files) if exists(n)]
        if ready:
            emit({"type": "text", "route": "sheep", "text": "Your files are ready in the workspace: " + ", ".join(ready)
                  + ". Say pptx, docx or xlsx for a Microsoft copy."})
            return finish()

    if job and not exec_calls and executor is local and (orch.write_mode(job["kind"]) if orch else True):
        return shepherd_job(job, prev_answer if job["reuse"] else "")

    if exec_calls:
        done_lines = []
        for i, c in enumerate(exec_calls):
            if sess.stop.is_set():
                return finish(stopped=True)
            cid = f"plan_{i}"
            emit({"type": "tool_call", "id": cid, "name": c["tool"], "args": c["args"], "route": "sheep"})
            ok, result = call(cid, c["tool"], dict(c["args"]), "sheep")
            if sess.stop.is_set():
                return finish(stopped=True)
            if not ok:
                # one call failed: the local model finishes the job, told what is done
                convo.append({"role": "system", "content": "Already done, do not repeat:\n" + "\n".join(done_lines)
                              + f"\nThis call failed: {c['tool']} {json.dumps(c['args'])[:300]}: {result}\n"
                              "Fix that one step, then finish the job."})
                break
            done_lines.append(f"- {c['tool']}: {result[:120]}")
        else:
            if sess.stop.is_set():
                return finish(stopped=True)
            if ensure_explicit_speech_output():
                return finish(stopped=sess.stop.is_set())
            deck = next((c["args"].get("path") for c in exec_calls if c["tool"].startswith("slides_") and c["args"].get("path")), "")
            if job and job.get("sources") and deck and not any(
                    (t or "").strip().lower() == "sources" for t, _ in shepherd.deck_slides(ws, deck)):
                # a deck written from the web ends with where its facts came from, even when the plan forgot
                call("plan_sources", "slides_add", {"path": deck, "title": "Sources", "bullets": [
                    f"{r.get('title') or r['url']}: {r['url']}" for r in job["sources"][:5]]}, "sheep")
            slides = sum(c["tool"] == "slides_add" for c in exec_calls)
            emit({"type": "text", "route": "cloud", "text": summary() + (f" {slides + 1} slides, written by Nemotron and "
                  "made on this machine." if slides else " Written by Nemotron, made on this machine.")})
            return finish()

    # one exact command ("render intro.sstudio", "copy a.txt to b.txt"): Sheep runs the
    # tool itself and the job ends (direct.py says why). It still asks the person's yes.
    if not attached and perm != "plan":
        from .. import direct
        cmd = direct.parse(asked)          # exact words: file names in a one-line command must not change
        if cmd:
            tool, dargs, words = cmd
            if T.TOOLS[tool].tier > tier:
                emit({"type": "text", "route": "local", "text":
                      f"This model may create and edit files but not {tool.split('_')[0]} them yet: that opens once it "
                      "passes the Checker exam (Settings › Advanced › Checker). Nothing was changed."})
                return finish()
            cid = "direct_" + uuid.uuid4().hex[:6]
            emit({"type": "tool_call", "id": cid, "name": tool, "args": dargs, "route": "sheep"})
            ok, result = call(cid, tool, dict(dargs), "sheep")
            if sess.stop.is_set():
                return finish(stopped=True)
            if ok:
                said = words or (result[:1].upper() + result[1:]).rstrip(".") + "."
                emit({"type": "text", "route": "sheep", "text": said})
                return finish()
            if "declined" in result:
                emit({"type": "text", "route": "sheep", "text": "Nothing was changed."})
                return finish()
            # it did not work as asked: the model takes the job, knowing why
            convo.append({"role": "system", "content": f"Sheep tried {tool} {json.dumps(dargs)} and it failed: {result}. "
                                                       "Fix that with the tools, or tell the person why it cannot be done."})

    fitted = 0
    for step in range(MAX_STEPS):
        if sess.stop.is_set():
            return finish(stopped=True)
        # a hot machine cools between local steps (bounded); a cloud step does not heat it
        if executor is not cloud and orch is not None and getattr(orch, "thermal", None):
            orch.thermal.pace(emit, sess.stop)
            if sess.stop.is_set():
                return finish(stopped=True)
        emit({"type": "thinking", "route": "cloud" if executor is cloud else "local", "step": step + 1,
              "what": "Nemotron is working" if executor is cloud else "Sheep is working"})
        try:
            kw = {"why": "step the local model could not do"} if executor is cloud else base_kw()
            if executor is not cloud and not schemas:          # a plain answer streams as it is written
                kw["on_delta"] = lambda piece: emit({"type": "draft", "delta": piece})
            msg, info = executor.chat(convo, schemas or None, **kw)
        except LLMError as e:
            if too_long(e) and fitted < 3:
                # a big tool result (a full folder, a long file) ran the prompt over:
                # shorten and ask again, never show the engine's error
                fitted += 1
                convo[:] = fit(convo, fitted)
                emit({"type": "route", "route": "local", "why": "the conversation was shortened to fit the model"})
                continue
            if executor is local and cloud_on:
                executor = cloud
                emit({"type": "route", "route": "cloud", "why": f"local engine failed ({e}); escalated"})
                continue
            if too_long(e):
                # said in words, not as the engine's 400 (2026-10-13 simulation)
                emit({"type": "text", "route": "local", "text": "This chat has grown longer than the model can hold. "
                      "Start a New chat for this request; your files are all still in the workspace."})
                return finish()
            emit({"type": "error", "text": error_words(e, local=executor is local)})
            return finish()
        emit({"type": "step", "n": step + 1, "route": info["route"], "ms": round(info["ms"]),
              "tokens": info.get("tokens", {})})
        if executor is local and orch:
            orch.after_turn(info)
            if info.get("degenerate"):
                # the guard cut the reply: never show it; Nebius takes the turn when it can
                if cloud_on:
                    executor = cloud
                    stand_in = True
                    emit({"type": "route", "route": "cloud", "why": "the reply fell into a loop; Nebius answers this turn"})
                    continue
                emit({"type": "text", "route": "local", "text": "That reply fell into a loop, so it was stopped and "
                      "not shown. The orchestrator is re-checking the model; try again in a moment."})
                return finish()
        calls = msg.get("tool_calls") or []
        text = (msg.get("content") or "").strip()
        if text.startswith("<|python_tag|>"):        # Llama 3.2's marker for a call, left in the text
            text = text[len("<|python_tag|>"):].strip()
        if not calls and _looks_like_call(text):
            fixed = _repair_call(text, set(T.TOOLS))
            if fixed:
                calls = [fixed]
                text = ""
        if not calls and text.startswith("{") and '"name"' in text[:40] and step < MAX_STEPS - 1:
            # a tool call written as text the engine could not read (bad JSON):
            # never shown as an answer; the model is told and tries again
            emit({"type": "tool_result", "id": f"bad{step}", "name": "(tool call)", "ok": False,
                  "text": "the call was not valid JSON: " + text[:240], "route": info["route"]})
            convo.append({"role": "assistant", "content": text})
            convo.append({"role": "user", "content": "That tool call was not valid JSON. Call the tool again with "
                                                     "valid JSON arguments, or reply to me in plain words if the job is done."})
            fails += 1
            if fails >= 3:
                emit({"type": "text", "text": "I could not complete this: my tool calls were not valid.", "route": "local"})
                return finish(failed=True)
            continue
        if not calls and executor is local:
            # The checker on a finished local reply: template leaks trimmed, and an
            # empty reply, a loop, or JSON/code that does not parse goes back once.
            # Structure only; what the model says is the person's model's to say.
            text = checker.tidy(text)
            faults = checker.live_faults(text, asked)
            if faults and not rechecked and step < MAX_STEPS - 1:
                rechecked = True
                emit({"type": "draft", "delta": None})            # the faulty draft leaves the screen
                emit({"type": "thinking", "route": "local", "what": "Checking the reply: " + faults[0]})
                convo.append({"role": "assistant", "content": text or "(empty)"})
                convo.append({"role": "user", "content": "Your reply had a fault: " + "; ".join(faults)
                              + ". Answer again, complete and correct."})
                continue
            if faults and ("empty reply" in faults or "a repeating loop" in faults):
                if cloud_on:
                    executor, stand_in = cloud, True
                    emit({"type": "route", "route": "cloud", "why": "the reply " + faults[0] + "; Nebius answers this turn"})
                    continue
                emit({"type": "text", "route": "local", "text": "That reply fell into a loop, so it was stopped and "
                      "not shown. The orchestrator is re-checking the model; try again in a moment."})
                return finish()
            if faults:
                text += "\n\n(Checked by Sheep: " + "; ".join(faults) + ".)"
        if not calls and stand_in and checker.refusal(text):
            # the stand-in would undo the person's own model: say what happened instead
            emit({"type": "text", "route": "local", "text": "Your model's reply fell into a loop, so it was not shown, "
                  "and the cloud stand-in declined this one. Try again in a moment; the orchestrator is re-checking "
                  "your model."})
            return finish()
        if not calls:
            # A chat model can describe a requested voice file without calling
            # speak_text (and may even invent an output path). Complete explicit
            # speech-file requests deterministically, and report only the path
            # actually written by the local voice tool.
            if (explicit_speech_output(asked)
                    and not any(Path(p).suffix.lower() == ".wav" for p in rc.made)):
                if "speak_text" not in names:
                    emit({"type": "text", "route": "local", "text":
                          "I could not create the recording: Sheep’s Windows voice tool is not available in this setup."})
                    return finish()
                try:
                    voice_path = speech_output_path(ws, asked)
                    scope_asked = asked + "\nRequested audio output: " + voice_path
                except ToolError as e:
                    emit({"type": "text", "route": "local", "text": f"I could not choose a safe audio output path: {e}"})
                    return finish()

                # Use explicit quoted text when supplied; otherwise write one
                # grounded sentence from the user's recent prompts. The voice
                # model never receives or uploads a photo.
                spoken = words_to_speak(asked)
                if not spoken:
                    prior = "\n".join(str(m.get("content") or "") for m in messages
                                       if m.get("role") == "user")[-1800:]
                    try:
                        emit({"type": "thinking", "route": "local", "what": "Writing a short voice line"})
                        line_msg, _ = local.chat(
                            [{"role": "system", "content":
                              "Write exactly one short, natural sentence for a Windows voice to read. "
                              "Use only subjects and details the user described in their messages. "
                              "Do not claim to see an image. Return only the sentence."},
                             {"role": "user", "content": prior}], None, max_tokens=48,
                            **base_kw(cache=False))
                        spoken = _clean_spoken_line(line_msg.get("content"))
                    except Exception:
                        spoken = _clean_spoken_line("")

                cid = "voice_" + uuid.uuid4().hex[:8]
                args = {"text": spoken, "path": voice_path}
                emit({"type": "tool_call", "id": cid, "name": "speak_text", "args": args, "route": "local"})
                called_tools.add("speak_text")
                ok, result = call(cid, "speak_text", args, "local")
                if ok:
                    actual = sorted(p for p in rc.made if Path(p).suffix.lower() == ".wav")
                    emit({"type": "text", "route": "local", "text":
                          "Created the spoken audio: " + ", ".join(actual) + "."})
                else:
                    emit({"type": "text", "route": "local", "text": f"I could not create the recording: {result}"})
                return finish()

            if (not rc.made and not acted and tier >= 2 and perm != "plan" and not cant_told
                    and re.search(CANT, text or "", re.I)):
                # It says it cannot do what Sheep can (the dog picture, 2026-10-14: "I don't have the
                # capability to generate images" with a picture model installed). Once, it is given
                # the tool that does it, told so, and asked again.
                need = next((t for pat, t, _ in MEDIA_NEEDS if re.search(pat, cue, re.I)), None)
                hint = None
                if need and able(state, [need])[0]:
                    if need not in names:
                        names = list(names) + [need]
                        schemas = T.schemas(names)
                    hint = f"Sheep can do this on this computer: call the {need} tool now."
                elif names:
                    hint = "Sheep can do this with the tools you have: call the one that does it now."
                if hint:
                    cant_told = True
                    emit({"type": "thinking", "route": "local", "what": "The model said it could not; Sheep can"})
                    convo.append({"role": "assistant", "content": text})
                    convo.append({"role": "user", "content": hint})
                    continue
            if not rc.made and not acted and tier >= 2 and perm != "plan" and re.search(CLAIM, text or "", re.I):
                # it says it made or changed a file and no tool ran (Ernest's dog-breed
                # deck, 2026-10-06): never shown as done. What it wrote goes in the file.
                j2 = job or shepherd.intent(asked + " put that", app, path, last_files, text, exists)
                if j2 and shepherd.structured(text):
                    return shepherd_job(dict(j2, reuse=True), text)
                if (job or names) and not claimed:
                    claimed = True
                    convo.append({"role": "assistant", "content": text})
                    convo.append({"role": "user", "content": "Nothing was changed: no tool was called. Call the tool "
                                                             "that does it now."})
                    continue
                if names:
                    text = "I could not do this with the tools, so nothing was changed. Try saying it another way."
                    gave_up = True
            if (tool_fails and not tool_oks and not rc.made and not acted
                    and not re.search(r"\b(could ?n[o']t|cannot|can't|unable|failed|not able|sorry|no tool)\b",
                                      text or "", re.I)):
                # every step failed and nothing changed, yet the reply says it is done ("The title size
                # has been increased", "saved as meeting.wav": both refused, 2026-10-14 live run)
                text = ("I could not do this, so nothing was changed. The last step said: "
                        + tool_fails[-1].split(" (use ")[0].rstrip(". ") + ".")
                gave_up = True
            # the work is checked against the request before it is called done:
            # one chance to fix what is short, then the person is told
            short = verify.shortfalls(ws, asked, rc.made) if rc.made else []
            if short and not checked and step < MAX_STEPS - 1:
                checked = True
                emit({"type": "thinking", "route": "local", "what": "Checking the work: " + short[0]})
                convo.append({"role": "assistant", "content": text or "Done."})
                convo.append({"role": "user", "content": "Checked against my request: " + "; ".join(short)
                              + ". Fix that with the tools, then tell me when it is done."})
                continue
            if short:
                text = (text or summary()) + "\n\nNot yet as asked: " + "; ".join(short) + "."
            emit({"type": "text", "text": text, "route": info["route"]})
            return finish(failed=gave_up)
        if text:
            emit({"type": "text", "text": text, "route": info["route"], "partial": True})
        convo.append({"role": "assistant", "content": text or None, "tool_calls": calls})
        step_failed = False
        for c in calls:
            fn = c.get("function", {})
            name = fn.get("name", "")
            called_tools.add(name)
            cid = c.get("id") or ("call_" + uuid.uuid4().hex[:8])
            try:
                args = _args(fn.get("arguments"))
                if name == "make_video":
                    if missing_video_attachments and not re.search(VIDEO_TEXT_ONLY, asked, re.I):
                        raise ToolError("an attached picture is no longer available in the workspace; reattach it before making this video")
                    _video_call_sources(ws, args, selected_video_images, asked)
                # media follow-up: "add music to that", "make it longer" — resolve to the last
                # .sstudio or .mp4 in this chat (plan A7). The request names no file.
                MEDIA_EXTS = (".sstudio", ".mp4")
                if (not args.get("path") and name.startswith("studio_")
                        and not named and last_files and re.search(r"\b(it|that|this|the video|the clip)\b", asked, re.I)):
                    media = [f for f in reversed(last_files) if f.endswith(MEDIA_EXTS)]
                    if media:
                        args["path"] = media[0]
                # parse seconds from the request for make_video: "a 4-second clip of ..."
                if name == "make_video" and "seconds" not in args:
                    m = re.search(r"\b(\d{1,2})[- ]?second", asked, re.I)
                    if m:
                        args["seconds"] = int(m.group(1))
                # the request named the file: a call aimed at another file of that kind goes to it
                p = str(args.get("path") or "")
                if p and named and p not in named and p not in rc.made and p != path:
                    same = [n for n in named if n.rsplit(".", 1)[-1].lower() == p.rsplit(".", 1)[-1].lower()]
                    if len(same) == 1:
                        args["path"] = same[0]
            except ToolError as e:
                args = {}
                emit({"type": "tool_call", "id": cid, "name": name, "args": {}, "route": info["route"]})
                emit({"type": "tool_result", "id": cid, "name": name, "ok": False, "text": str(e)})
                ok, result = False, str(e)
            else:
                key = name + json.dumps(args, sort_keys=True)
                if key in seen_calls and rc.made:
                    # the same call again after the work is done: a loop, not progress
                    emit({"type": "text", "text": summary(), "route": "local"})
                    return finish()
                seen_calls.add(key)
                content = _content(name, args)
                if content and content <= added and rc.made:
                    # the same content again: the job already did this
                    emit({"type": "text", "text": summary(), "route": "local"})
                    return finish()
                emit({"type": "tool_call", "id": cid, "name": name, "args": args, "route": info["route"]})
                if name not in names:
                    ok, result = False, f"'{name}' is not one of your tools. Your tools: {', '.join(names)}"
                    emit({"type": "tool_result", "id": cid, "name": name, "ok": False, "text": result})
                else:
                    ok, result = call(cid, name, args, info["route"])
                    if ok:
                        added.update(content)
                    # A request for a picture, a recording or a transcript and nothing
                    # else is done when that tool succeeds: the 3B made the picture and
                    # then wandered into decks and herders until it reported failure
                    # (2026-10-06). The orchestrator ends it, as it ends exact arithmetic.
                    rest = re.sub(AI_VIDEO, " ", asked.lower()) if name == "make_video" else asked.lower()
                    if ok and name in MEDIA_DONE and not re.search(OTHER_WORK, rest):
                        emit({"type": "text", "text": MEDIA_DONE[name](result), "route": "local"})
                        return finish()
                    if not ok and result.endswith("The job is done when the request is done.") and rc.made:
                        emit({"type": "text", "text": summary(), "route": "local"})
                        return finish()
            step_failed = step_failed or not ok
            (tool_oks if ok else tool_fails).append(result)
            convo.append({"role": "tool", "tool_call_id": cid, "content": (result + DONE_HINT) if ok else f"ERROR: {result}"})
        fails = fails + 1 if step_failed else 0
        if fails >= 2:
            if executor is local and cloud_on:
                executor = cloud
                schemas = T.schemas(able(state, T.offered(1 if perm == "plan" else 3, apps))[0])
                names = [s["function"]["name"] for s in schemas]
                emit({"type": "route", "route": "cloud", "why": "two failed steps in a row; escalated to Nebius"})
                fails = 0
            elif fails >= 3:
                emit({"type": "text", "text": "I could not complete this: the last three steps failed. "
                                              "See the tool results above.", "route": "local"})
                return finish(failed=True)
    emit({"type": "text", "text": f"Stopped after {MAX_STEPS} steps.", "route": "local"})
    return finish()
