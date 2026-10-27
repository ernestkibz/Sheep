"""The Checker: its rules on known faults, and a judge model installed when it
passes its exam and rolled back when it does not.

    python -m mlmi.tests.test_checker [JUDGE.ssf]

The judge defaults to ~/.mlmi/models/Qwen2.5-1.5B-Instruct.ssf; without one the
rules are still tested and the install/rollback part is skipped, said out loud.
"""
import sys
import time
from pathlib import Path

from mlmi.backend import checker as K
from mlmi.backend.modes import Local
from mlmi.backend.server import Engine

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def test_rules():
    r = K.rules
    check("a clean answer has no faults", r("Paris is the capital of France.", "capital?") == [])
    check("an empty reply", r("", "hi") == ["empty reply"])
    check("a tool call alone is not empty", r("", "weather?", [{"function": {"name": "w", "arguments": "{}"}}]) == [])
    check("leaked role markers", any("role markers" in f for f in r("Sure <|eot_id|> assistant", "x")))
    check("a repeating loop", any("loop" in f for f in r("the cat sat " * 12, "story")))
    check("a normal long answer is not a loop",
          r("Sleep restores the body, consolidates memory, balances hormones and repairs tissue; "
            "without it attention, mood and immunity all suffer within days.", "why sleep") == [])
    check("JSON asked for, prose given", any("JSON" in f for f in r("Here you go: name is Ann", "Return only valid JSON")))
    check("JSON asked for and given", r('{"name": "Ann"}', "Return only valid JSON") == [])
    check("Python that does not parse", any("Python" in f for f in r("```python\ndef f(:\n  pass\n```", "code")))
    check("Python that parses", r("```python\ndef f(x):\n    return x * 2\n```", "code") == [])
    check("tool-call arguments that are not JSON",
          any("arguments" in f for f in r("", "x", [{"function": {"name": "w", "arguments": "{city: Kampala"}}])))
    check("a web answer without sources", any("sources" in f for f in r("Prices rose 3%.", "prices?", sources=["https://a"])))
    check("a web answer citing [1]", r("Prices rose 3% [1].", "prices?", sources=["https://a"]) == [])


def run_judge(path):
    cfg = {}
    saves = []
    c = K.Checker(cfg, Engine, Local, lambda: saves.append(dict(cfg)))
    c.install(str(path))
    t0 = time.time()
    while c.trial["state"] == "running" and time.time() - t0 < 600:
        time.sleep(1)
    t = c.trial
    check("a good judge passes its exam and is installed", t["state"] == "installed",
          f"{t.get('right')}/{t.get('of')} in {t.get('seconds')} s")
    check("the installed judge is saved in the config", (cfg.get("checker") or {}).get("path") == str(path) and saves)
    faults, verdict = c.check("What is the capital of Uganda?", "Kampala.")
    check("the judge passes a right answer", not faults and verdict == "PASS", verdict)
    faults, verdict = c.check("What is the capital of Uganda?", "Nairobi is the capital of Uganda.")
    check("a learned negative verdict does not rewrite or block the answer", not faults and verdict == "FAIL", verdict)
    # a judge that cannot meet the mark is rolled back and the one before it stays
    mark = K.PASS_MARK
    K.PASS_MARK = 1.01
    try:
        c.install(str(path))
        t0 = time.time()
        while c.trial["state"] == "running" and time.time() - t0 < 600:
            time.sleep(1)
    finally:
        K.PASS_MARK = mark
    check("a judge below the mark is rolled back", c.trial["state"] == "rolled back", c.trial.get("why", ""))
    check("the judge before it stays active", c.judge is not None and (cfg.get("checker") or {}).get("path") == str(path))
    c.remove()
    check("back to the rules only", c.judge is None and "checker" not in cfg)


def main():
    test_rules()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".mlmi" / "models" / "Qwen2.5-1.5B-Instruct.ssf"
    if path.is_file():
        run_judge(path)
    else:
        print(f"  SKIP  the judge part: no {path}")
    bad = RESULTS.count(False)
    print(f"all {len(RESULTS)} checks passed" if not bad else f"{bad} of {len(RESULTS)} checks FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
