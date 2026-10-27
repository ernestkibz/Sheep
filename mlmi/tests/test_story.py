"""What Sheep did for you: the setup timeline reads the registry and says each step in plain words.

    python -m pytest mlmi/tests/test_story.py -q
"""
from mlmi.backend.story import story


def test_gguf_model_story_in_order_and_plain():
    entry = {"name": "bartowski--Qwen.ssf", "added": 1,
             "built": {"t": 2, "from": "GGUF", "level": "balanced", "minutes": 4, "origin": "bartowski/Qwen-GGUF"},
             "history": [{"t": 1, "event": "added", "note": "qwen2, 28 layers"},
                         {"t": 3, "event": "loaded"},
                         {"t": 4, "event": "canaries", "passed": 8, "of": 8},
                         {"t": 5, "event": "oriented", "passed": 14, "of": 14}]}
    facts = {"arch": "qwen2", "bytes": 2.1e9, "n_layer": 28}
    plan = {"device": "hybrid", "why": ["14 of 28 layers fit the GPU"]}
    words = [s["what"] for s in story(entry, facts, plan)]
    assert words[0] == "Added bartowski/Qwen"
    assert any("GGUF" in w for w in words) and any(w.startswith("Built it at the balanced level") for w in words)
    assert "Placed it on the graphics card and the processor together, sized for this computer" in words
    assert words[-2:] == ["Checked its answers: 8 of 8 right", "Checked app tasks: 14 of 14 work"]
    for w in words:
        assert "layer" not in w and "ssf" not in w


def test_compat_and_upgrade_and_repair():
    entry = {"name": "x.compat.json", "built": {"level": "compat"},
             "history": [{"t": 1, "event": "added"}, {"t": 6, "event": "egypt: returned", "cause": "a shorter context"},
                         {"t": 7, "event": "upgrade: started", "source": "x"}]}
    words = [s["what"] for s in story(entry, {"compat": True}, {"device": "compat"})]
    assert any("maker's own code" in w for w in words)
    assert "Repaired it: a shorter context" in words
    assert any(w.startswith("Moved it from the maker's code") for w in words)


def test_no_model_no_story():
    assert story(None) == []
