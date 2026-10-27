"""Sheep + Nebius, the cloud half, against the real cloud endpoint.

    python mlmi/tests/test_blend.py      # needs NEBIUS_API_KEY (and NEBIUS_BASE_URL / NEBIUS_MODEL)

1. The cloud answers (NVIDIA Nemotron).
2. The Egypt flight: while the local model is being fixed, a turn in Sheep +
   Nebius mode is answered in the cloud and never by the faulty model.
3. In Sheep mode the same turn is held back, and nothing leaves the machine.
"""
import os
import sys
import tempfile
import threading
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mlmi.backend import agent  # noqa: E402
from mlmi.backend.modes import Cloud  # noqa: E402
from mlmi.backend.workspace import Workspace  # noqa: E402


class Holding:
    """An orchestrator in the middle of fixing a model."""
    def holding(self):
        return True

    def chat_kwargs(self):
        return {}

    def after_turn(self, meta):
        pass


class NoLocal:
    """A faulty local model: calling it at all is a failure of the test."""
    def chat(self, *a, **k):
        raise AssertionError("the faulty local model was asked to answer")


def state(mode, cloud):
    return types.SimpleNamespace(
        session=agent.Session(), ws=Workspace(Path(tempfile.mkdtemp(prefix="mlmi-blend-"))),
        model_card=lambda: {"tier": 2}, mode=mode, permission="auto", local=NoLocal(), cloud=cloud, orch=Holding(),
        busy=threading.Lock())


def main():
    cloud = Cloud()
    assert cloud.ready(), "no NEBIUS_API_KEY in the environment"
    print(f"cloud: {cloud.model} via {cloud.provider}")
    msg, info = cloud.chat([{"role": "user", "content": "Reply with the single word: ready"}], why="test")
    print("  PASS  the cloud answers:", (msg.get("content") or "").strip()[:40], f"({info['ms'] / 1000:.1f} s)")

    ev = []
    agent.run(state(1, cloud), [{"role": "user", "content": "What is the capital of Uganda? One word."}], emit=ev.append)
    routes = [e for e in ev if e["type"] == "route"]
    texts = [e["text"] for e in ev if e["type"] == "text" and e.get("text")]
    assert routes and routes[0]["route"] == "cloud", ev[:3]
    assert texts and "kampala" in texts[-1].lower(), texts
    print("  PASS  Egypt flight: the cloud answered while the model was fixed:", texts[-1][:60])
    sent = len(cloud.sent)

    ev = []
    agent.run(state(2, cloud), [{"role": "user", "content": "What is the capital of Uganda? One word."}], emit=ev.append)
    texts = [e["text"] for e in ev if e["type"] == "text"]
    assert texts and "fixing" in texts[-1].lower() and len(cloud.sent) == sent, texts
    print("  PASS  Sheep mode: the reply was held back and nothing was sent")
    print("all 3 checks passed")


if __name__ == "__main__":
    main()
