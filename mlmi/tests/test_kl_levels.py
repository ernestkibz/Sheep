"""KL scorer regression tests independent of model checkpoints."""
import unittest

from tooling.sml_kl_levels import kl_from_logits, _level_id


class KLDivergenceTests(unittest.TestCase):
    def test_identical_logits_have_zero_kl(self):
        self.assertEqual(kl_from_logits([1.0, 2.0, -1.0], [1.0, 2.0, -1.0]), 0.0)

    def test_sequence_kl_averages_each_prefix_distribution(self):
        first = kl_from_logits([[2.0, 0.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]])
        one_changed_row = kl_from_logits([2.0, 0.0], [0.0, 2.0])
        self.assertAlmostEqual(first, one_changed_row / 2.0)

    def test_kl_is_nonnegative_and_directional(self):
        p = [4.0, 0.0, -2.0]
        q = [1.0, 3.0, -1.0]
        self.assertGreater(kl_from_logits(p, q), 0.0)
        self.assertNotAlmostEqual(kl_from_logits(p, q), kl_from_logits(q, p))

    def test_vocabulary_mismatch_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "vocabularies differ"):
            kl_from_logits([1, 2], [1])

    def test_quantization_level_names_map_to_project_recipes(self):
        self.assertEqual(_level_id("Q4"), "small")
        self.assertEqual(_level_id("Q4/6"), "balanced")
        self.assertEqual(_level_id("Q6"), "high")


if __name__ == "__main__":
    unittest.main()
