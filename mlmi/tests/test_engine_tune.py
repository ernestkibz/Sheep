"""The engine tune (2026-10-26): each model tries a few GPU launch settings on this machine and keeps
one only when its answer is word for word the same and it wins every paired race. Tunes are keyed
by the model's technical fingerprint from its header, never by its file name."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import orchestrator as O  # noqa: E402

QWEN = {"arch": "qwen2", "n_layer": 36, "n_embd": 2048, "n_head": 16, "n_kv": 2, "head_dim": 128,
        "n_weights": 3_000_000_000, "widths": [4, 6], "exact": False}


class FakeEngine:
    def __init__(self, speed, text):
        self.device, self.tune_env, self.compat_running = "gpu", {}, False
        self.speed, self.text, self.starts = speed, text, []

    def stop(self):
        pass

    def start(self, path):
        self.starts.append(dict(self.tune_env))
        return True


class FakeLocal:
    def __init__(self, eng):
        self.eng = eng

    def chat(self, *a, **k):
        env = " ".join(f"{k}={v}" for k, v in self.eng.tune_env.items())
        return {"content": self.eng.text(env)}, {}

    def stats(self):
        env = " ".join(f"{k}={v}" for k, v in self.eng.tune_env.items())
        return {"last_prompt_ms": 0, "last_gen_ms": 1000 / self.eng.speed(env)}


def orch(speed, text=lambda env: "same answer"):
    eng = FakeEngine(speed, text)
    st = SimpleNamespace(engine=eng, local=FakeLocal(eng), model_path="m.ssf")
    said = []
    o = SimpleNamespace(st=st, facts=dict(QWEN), stage="", freed=lambda d: d,
                        _background_yield_reason=lambda: None, say=lambda *a: said.append(a[0]))
    return o, eng, said


class EngineTune(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = [patch.object(O, "TUNES", Path(self.tmp.name) / "engine_tunes.json"),
                  patch.object(O, "probe", lambda *a: {}), patch.object(O, "engine_stamp", lambda: "e1"),
                  patch.object(O, "hardware_sig", lambda d: "rtx3050|ac")]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        self.tmp.cleanup()

    def test_fingerprint_ignores_the_name(self):
        self.assertEqual(O.tech_sig(dict(QWEN, source="mistral-named-file")), O.tech_sig(QWEN))
        self.assertNotEqual(O.tech_sig(dict(QWEN, arch="llama")), O.tech_sig(QWEN))

    def test_a_clear_winner_is_kept_and_used(self):
        o, eng, said = orch(lambda env: 120.0 if env == "SML_GRID=64" else 100.0)
        self.assertTrue(O.Orchestrator.engine_tune(o, "m.ssf"))
        self.assertEqual(eng.tune_env, {"SML_GRID": "64"})
        key = f"{O.tech_sig(QWEN)}#rtx3050|ac"
        self.assertEqual(O.load_tunes()[key]["env"], {"SML_GRID": "64"})
        self.assertIn("faster", said[-1])
        # the same model under another file name finds the same tune
        o.facts = dict(QWEN)
        o.st.model_path = "renamed-llama.ssf"
        self.assertEqual(O.Orchestrator.tuned_env(o, "renamed-llama.ssf", "gpu", {}), {"SML_GRID": "64"})

    def test_a_faster_setting_that_changes_the_answer_is_dropped(self):
        o, eng, said = orch(lambda env: 150.0 if env == "SML_GRID=64" else 100.0,
                            text=lambda env: "different" if env == "SML_GRID=64" else "same answer")
        O.Orchestrator.engine_tune(o, "m.ssf")
        rec = next(iter(O.load_tunes().values()))
        self.assertEqual(rec["env"], {})
        self.assertIn("SML_GRID=64", rec["rejected"])
        self.assertEqual(eng.tune_env, {})

    def test_a_gain_inside_the_noise_is_not_kept(self):
        o, eng, said = orch(lambda env: 101.0 if env == "SML_GRID=64" else 100.0)
        O.Orchestrator.engine_tune(o, "m.ssf")
        self.assertEqual(next(iter(O.load_tunes().values()))["env"], {})
        self.assertIn("already the fastest", said[-1])

    def test_the_status_returns_to_in_sync(self):
        o, eng, said = orch(lambda env: 100.0)
        O.Orchestrator.engine_tune(o, "m.ssf")
        self.assertEqual(o.stage, "in sync")

    def test_runs_once_per_engine_build(self):
        o, eng, said = orch(lambda env: 100.0)
        O.Orchestrator.engine_tune(o, "m.ssf")
        n = len(eng.starts)
        O.Orchestrator.engine_tune(o, "m.ssf")
        self.assertEqual(len(eng.starts), n)

    def test_big_cards_also_try_wider_grids(self):
        grids = lambda gb: [c["SML_GRID"] for c in O.tune_candidates({"gpu": {"vram_total": gb * 2**30}}) if "SML_GRID" in c]
        self.assertEqual(grids(4), ["32", "64", "96"])          # the laptop card: nothing wider
        self.assertIn("128", grids(8))
        self.assertIn("512", grids(80))                          # an H100

    def test_cpu_models_are_left_alone(self):
        o, eng, said = orch(lambda env: 100.0)
        eng.device = "cpu"
        self.assertTrue(O.Orchestrator.engine_tune(o, "m.ssf"))
        self.assertEqual(eng.starts, [])


if __name__ == "__main__":
    unittest.main()
