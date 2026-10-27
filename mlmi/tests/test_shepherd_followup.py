import unittest

from mlmi.backend.shepherd import calls, intent, parse, reuse_needs_reformat, writer_prompt


class FollowupArtifactTests(unittest.TestCase):
    def test_put_prose_answer_in_document_reuses_previous_answer(self):
        answer = "Here are five useful tips: stay focused, take breaks, and plan ahead."
        job = intent("put that in a document", prev_answer=answer)
        self.assertIsNotNone(job)
        self.assertEqual(job["kind"], "words")
        self.assertEqual(job["action"], "create")
        self.assertTrue(job["reuse"])
        parsed = parse("words", answer)
        self.assertTrue(parsed[1])
        self.assertTrue(reuse_needs_reformat("words", answer, parsed))
        self.assertEqual(calls(job, parsed)[0][0], "words_create")
        self.assertEqual(calls(job, parsed)[1][0], "words_add")

    def test_reformat_prompt_carries_original_request_and_answer(self):
        answer = "Here are five football tips in one paragraph."
        job = intent("put that in a document", prev_answer=answer)
        prompt = writer_prompt({**job, "source_request": "give me five tips about football tactics",
                                "source_answer": answer})
        self.assertIn("Original request: give me five tips about football tactics", prompt)
        self.assertIn("Previous assistant answer:\n" + answer, prompt)

    def test_structured_reused_answer_does_not_need_reformatting(self):
        answer = "# Tips\n\n- Focus on shape\n- Press together\n- Protect the center\n- Switch play\n- Manage transitions"
        parsed = parse("words", answer)
        self.assertFalse(reuse_needs_reformat("words", answer, parsed))

    def test_put_answer_in_document_without_previous_answer_does_not_invent_reuse(self):
        job = intent("put that in a document")
        self.assertIsNotNone(job)
        self.assertFalse(job["reuse"])


if __name__ == "__main__":
    unittest.main()
