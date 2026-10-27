"""The Egypt protocol against the live engine: a fault is forced (sampling at
temperature 3, which makes any model babble), and the orchestrator must sense
it (canaries), flee (hold replies in Sheep mode), find the cause (the doctor
tries the chat frame, then the model's own sampling), and return.

    python mlmi/tests/test_egypt.py            # needs ML.M.I's engine on :8099

It uses a throwaway registry key, so no real model's history is touched.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mlmi.backend import orchestrator as O  # noqa: E402
from mlmi.backend.modes import Cloud, Local  # noqa: E402

KEY = "__egypt_test__"


def main():
    st = types.SimpleNamespace(cfg={"orchestrator": True}, model_path=KEY, mode=2, stats={},
                               local=Local("http://127.0.0.1:8099"), cloud=Cloud())
    o = O.Orchestrator(st)
    o.facts = {"template": True, "stops": [1]}
    o.sampling = {"temperature": 3.0, "top_p": 1.0}       # the forced fault
    try:
        ok = o.canary_suite()
        e = o.egypt or {}
        print("canaries passed:", ok)
        print("stages:", [h["event"] for h in O.load_registry().get(KEY, {}).get("history", [])])
        print("sensed:", e.get("fails"))
        print("tried:", e.get("tried"))
        print("cause:", e.get("cause"), "| stage:", e.get("stage"), "| lamp:", o.lamp)
        print("never again:", O.load_registry().get(KEY, {}).get("never"))
        assert not ok, "the forced fault was not sensed"
        assert e.get("stage") == "returned" and e.get("cause") == "sampling", "the doctor did not fix it"
        assert o.lamp == "green" and o.sampling and o.sampling["temperature"] < 1
        print("PASS  test_egypt")
    finally:
        reg = O.load_registry()
        reg.pop(KEY, None)
        O.save_registry(reg)


if __name__ == "__main__":
    main()
