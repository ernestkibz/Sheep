"""Keys entered in Settings (hackathon B2): judges paste the Nebius and Tavily keys in the app instead of
setting environment variables; the key wins over the environment and is never shown back."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import server  # noqa: E402
from mlmi.backend.modes import Cloud  # noqa: E402


class KeysInSettings(unittest.TestCase):
    def test_config_key_wins_and_is_never_shown(self):
        env = {"NEBIUS_API_KEY": "from-env", "NEBIUS_BASE_URL": "https://openrouter.ai/api/v1"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("TAVILY_API_KEY", None)
            server.apply_keys({"keys": {"nebius_key": "from-settings", "tavily_key": "tv-1"}})
            self.assertEqual(os.environ["NEBIUS_API_KEY"], "from-settings")
            view = server.keys_view()
            self.assertEqual(view["nebius"], True)
            self.assertEqual(view["tavily"], True)
            self.assertNotIn("from-settings", repr(view))
            self.assertNotIn("tv-1", repr(view))

    def test_cloud_reads_the_key_again(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEBIUS_API_KEY", None)
            c = Cloud()
            self.assertFalse(c.ready())
            server.apply_keys({"keys": {"nebius_key": "k-123"}})
            c.reload()
            self.assertTrue(c.ready())

    def test_empty_values_change_nothing(self):
        with patch.dict(os.environ, {"NEBIUS_MODEL": "nvidia/nemotron-3-nano-30b-a3b"}, clear=False):
            server.apply_keys({"keys": {"nebius_model": "  "}})
            self.assertEqual(os.environ["NEBIUS_MODEL"], "nvidia/nemotron-3-nano-30b-a3b")


class FirstRunWithNemotron(unittest.TestCase):
    def test_welcome_offers_nemotron_and_the_chat_opens_without_a_local_model(self):
        ui = Path(__file__).resolve().parents[1] / "desktop" / "ui"
        welcome = (ui / "welcome.js").read_text(encoding="utf-8")
        app = (ui / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="wCloud"', welcome)
        self.assertIn('api("/api/settings", { mode: 1 })', welcome)
        self.assertIn('id="wKeys"', welcome)
        self.assertIn("!S.state.model.path && !cloudChat(S.state)", app)
        self.assertIn("const none = !st.model.path && !cloudChat(st);", app)


if __name__ == "__main__":
    unittest.main()
