import unittest

from mlmi.backend.quantizer import Job, recipe


class QuantizerAsIsStateTests(unittest.TestCase):
    def test_exact_selection_is_reported_as_a_distinct_job_state(self):
        job = Job()
        job.src = "org/model"
        job.state = "building"
        job.level = "exact"

        status = job.status()
        self.assertEqual(status["state"], "building")
        self.assertEqual(status["level"], "exact")
        self.assertEqual(status["src"], "org/model")

    def test_as_is_recipe_uses_exact_builder_without_quantization_options(self):
        command = recipe("source", "target.ssf", 4, licence="apache-2.0", level="exact")

        self.assertIn("--exact", command)
        self.assertIn("--license", command)
        self.assertIn("apache-2.0", command)
        self.assertNotIn("--alloc", command)
        self.assertNotIn("--layout", command)


if __name__ == "__main__":
    unittest.main()
