"""Guard the compact, accessible readiness summary against regressions."""
import unittest
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend.orchestrator import Orchestrator  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


class ReadinessPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ui = ROOT / "mlmi" / "desktop" / "ui"
        cls.app = (ui / "app.js").read_text(encoding="utf-8")
        cls.html = (ui / "index.html").read_text(encoding="utf-8")
        cls.css = (ui / "app.css").read_text(encoding="utf-8")

    def test_no_model_does_not_show_pending_check_counts(self):
        self.assertIn("No model selected. Add a model to start; checks haven’t begun.", self.app)
        self.assertIn("const showChecks = hasModel &&", self.app)
        self.assertIn('$("#rbChoose").hidden = hasModel;', self.app)
        self.assertIn('$("#statusText").textContent = "Add model";', self.app)
        self.assertIn('$("#status").className = "status add-model";', self.app)
        self.assertIn('openSettings(!S.state.model.path ? "models" : "dashboard")', self.app)
        self.assertIn("Sheep’s local service is still unavailable", self.app)

    def test_three_plain_rows_open_settings_activity(self):
        self.assertIn('readinessNames = { startup: "Model ready", canaries: "Answers checked", orientation: "Apps checked" }', self.app)
        self.assertIn('class="readiness-row" data-readiness-check=', self.app)
        self.assertIn('openSettings("dashboard", "chk-" + row.dataset.readinessCheck)', self.app)
        self.assertIn('dashboard: ["activity"]', self.app)      # the old page name opens Settings › Activity
        self.assertIn('function openSettings(tab = "basics", focus = "")', self.app)
        self.assertNotIn('id="rbIssue"', self.html)
        views = (ROOT / "mlmi" / "desktop" / "ui" / "views.js").read_text(encoding="utf-8")
        self.assertIn('sect("chk-" + c.id', views)
        self.assertIn("data-check-again", views)
        self.assertIn('action: "recheck"', views)

    def test_card_closes_and_stays_closed_until_a_check_changes(self):
        self.assertIn('id="rbClose" aria-label="Close"', self.html)
        self.assertIn('localStorage.setItem("sheep.readiness.closed", S.readinessFp', self.app)
        self.assertIn("const fp = `${st.model.path}|${checks.map((c) => `${c.id}:${c.state}`).join(\",\")}`;", self.app)
        self.assertIn('(!waiting && (allPassed || closedFp === fp))', self.app)

    def test_app_check_failure_keeps_chat_ready_and_shows_task_result(self):
        self.assertIn('$("#rbText").textContent = "Chat is ready";', self.app)
        self.assertIn('`${passed} of ${total} app tasks work`', self.app)
        self.assertIn('`Checking app tasks · ${ran} of ${total}`', self.app)
        self.assertNotIn("/${c.of || c.total} passed", self.app)          # never "0/14 passed" while checking
        self.assertIn('c.state === "failed" && c.id === "startup" ? "red"', self.app)
        self.assertIn('appFailures === 1 ? " needs" : "s need"', self.app)
        self.assertIn('if u.path == "/api/orient/retry"', (ROOT / "mlmi" / "backend" / "server.py").read_text(encoding="utf-8"))

    def test_startup_progress_is_indeterminate_until_a_result_is_measured(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.st = SimpleNamespace(model_path="test.ssf")
        orch.ready = lambda: False
        orch.loading = threading.Event()
        orch.loading.set()
        orch._startup_started = None
        orch._startup_elapsed = None
        orch.stage = "starting the engine"
        orch.progress = 0.1  # internal phase estimate, not measured completion
        orch.plan = {"device": "gpu", "device_name": "Test GPU"}
        orch.background_checks = {}
        startup = orch.readiness_checks()[0]
        self.assertEqual(startup["state"], "running")
        self.assertIsNone(startup["progress"])

    def test_waiting_check_does_not_display_stale_full_progress(self):
        self.assertIn("const hasCompletedWork = validNumber(c.done) && Number(c.done) > 0;", self.app)
        self.assertIn('const moving = c.state === "running" || String(c.state || "").startsWith("waiting for ");', self.app)
        # the heat guard reads CPU and GPU, so the label names neither (2026-10-12)
        self.assertIn('"waiting for temperature": "Paused: this computer is hot; it continues when it cools"', self.app)

    def test_no_model_status_is_neutral_and_steady(self):
        self.assertIn("--status-neutral:#f4f4f4", self.css)
        self.assertIn("--status-neutral:#171717", self.css)
        self.assertIn(".status.add-model .dot { background:var(--status-neutral)", self.css)
        self.assertIn("animation:none", self.css)
        self.assertIn('type="button" class="status" id="status"', self.html)

    def test_details_are_accessible_and_preference_is_saved(self):
        self.assertIn('aria-controls="rbDetails"', self.html)
        self.assertIn('aria-expanded="false"', self.html)
        self.assertIn('const READY_PREF = "sheep.readiness.expanded"', self.app)
        self.assertIn("localStorage.setItem(READY_PREF", self.app)
        self.assertIn('id="rbCancel"', self.html)
        self.assertIn('aria-label="Stop loading this model"', self.html)

    def test_dialog_close_is_not_force_and_hidden_modals_do_not_block_input(self):
        self.assertIn('button.onclick = () => close(false)', self.app)
        self.assertIn('This request is waiting for a compatible model.', self.app)
        self.assertIn("data-stop-pending", self.app)
        self.assertIn(".modal[hidden] { display:none !important; }", self.css)


if __name__ == "__main__":
    unittest.main()
