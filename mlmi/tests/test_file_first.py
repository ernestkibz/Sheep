"""Files first (Ernest's chat, 2026-10-20): "powerpoint about legend of seeker" got offers and no file, and the
short turns after it ("season 2", "your choice", "pptx") were read alone."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlmi.backend import shepherd as S  # noqa: E402


def U(t):
    return {"role": "user", "content": t}


def B(t, files=None):
    return {"role": "assistant", "content": t, **({"files": files} if files else {})}


class NounLed(unittest.TestCase):
    def test_a_file_named_by_kind_and_subject_is_made(self):
        for t in ("powerpoint about legend of seeker", "want a powerpoint about legend of seeker",
                  "a deck on water saving", "slides about dog breeds"):
            j = S.intent(t)
            self.assertIsNotNone(j, t)
            self.assertEqual((j["kind"], j["action"]), ("slides", "create"), t)

    def test_chat_asks_stay_chat(self):
        for t in ("read me a story about a lion", "tips for my presentation", "what is a powerpoint?"):
            self.assertIsNone(S.intent(t), t)


class FollowOn(unittest.TestCase):
    ask = [U("powerpoint about legend of seeker"), B("Sure! Would you like me to cover the first season?")]

    def test_detail_after_a_request_that_made_no_file(self):
        msgs = self.ask + [U("season 2")]
        self.assertEqual(S.follow_on(msgs), "powerpoint about legend of seeker — season 2")
        msgs += [B("Which characters should it focus on?"), U("your choice")]
        self.assertEqual(S.follow_on(msgs), "powerpoint about legend of seeker — season 2; your choice")

    def test_after_a_made_deck(self):
        made = [U("powerpoint about legend of seeker"), B("Made it.", files=["legend.sdeck"])]
        self.assertEqual(S.follow_on(made + [U("pptx")]), "export legend.sdeck to pptx")
        self.assertEqual(S.follow_on(made + [U("season 2")]), "add two slides about season 2 to legend.sdeck")
        for t in ("thanks", "yes", "i want a file", "what about season 3?"):
            self.assertIsNone(S.follow_on(made + [U(t)]), t)

    def test_format_word_finds_the_deck_a_few_turns_back(self):
        msgs = [U("powerpoint about legend of seeker"), B("Made it.", files=["legend.sdeck", "legend.pptx"]),
                U("i want a file"), B("Which file would you like?"), U("pptx")]
        self.assertEqual(S.follow_on(msgs), "export legend.sdeck to pptx")

    def test_nothing_to_follow(self):
        self.assertIsNone(S.follow_on([U("hey"), B("Hello! How can I assist you today?"), U("yes")]))
        self.assertIsNone(S.follow_on([U("yes")]))


if __name__ == "__main__":
    unittest.main()
