"""One chat session's state: stop requests, pending confirmations, the action in progress.

Split out of agent.py (2026-10-21); every name is re-exported from the agent package."""
import threading


class Session:
    """Shared by the agent runs of one server: confirmations waiting on a person, and a stop flag."""

    def __init__(self):
        self.pending = {}
        self.stop = threading.Event()
        self.action_lock = threading.Lock()
        self.active_action = ""
        self.active_cancel = None

    def reset_stop(self):
        with self.action_lock:
            self.active_action = ""
            self.active_cancel = None
            self.stop.clear()

    def set_cancel_hook(self, callback=None):
        """Register only a tool's safe, local cancellation hook."""
        with self.action_lock:
            self.active_cancel = callback
            already_stopping = self.stop.is_set() and callback is not None
        if already_stopping:
            try:
                callback()
            except Exception:
                pass

    def begin_action(self, name):
        with self.action_lock:
            if self.stop.is_set():
                return False
            self.active_action = name
            return True

    def end_action(self, name):
        with self.action_lock:
            if self.active_action == name:
                self.active_action = ""

    def request_stop(self):
        with self.action_lock:
            self.stop.set()
            action, callback = self.active_action, self.active_cancel
        if callback:
            try:
                callback()
            except Exception:
                pass
        return action

    def answer(self, cid, allow):
        p = self.pending.get(cid)
        if p:
            p["allow"] = bool(allow)
            p["event"].set()
            return True
        return False


def wait_for_model_selection(sess, cid, required, emit, timeout=600):
    """Hold a feature action until the UI confirms its checked local model."""
    ev = threading.Event()
    sess.pending[cid] = {"event": ev, "allow": False}
    emit({"type": "model_required", "id": cid, "slot": required.slot,
          "repo_id": required.repo_id, "message": str(required)})
    answered = ev.wait(timeout=timeout)
    answer = sess.pending.pop(cid, {})
    return answered and bool(answer.get("allow"))
