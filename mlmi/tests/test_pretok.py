"""A BPE tokenizer's own text pipeline in the engine (2026-10-19): the engine must cut text
into the same ids as the model's tokenizer.json. Tiny tokenizers are trained here with the
pipelines real models use (GPT-2 byte level, Llama 3 regex split, SmolLM2 digits, Falcon
punctuation, Mistral/Phi-3 Metaspace), packed as Sheep packs them, and cut by sml-cli."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tooling"))
CLI = ROOT / "sml" / "sml-cli.exe"

TEXT = ["Hello, world! The year 2026 costs $19.99 (or ½ of ² things).", "def f(x):\n    return x+1\n\n",
        "  leading and   inner spaces ", "Café naïve Straße 日本語 العربية", "a-b_c.d/e!!!...", "para one\n\npara two",
        "12345 67890 3.14159 -42 1,000,000", "emoji 🙂 🚀 and ¿Dónde? «quoted»"] * 3

LLAMA3 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
          r"\s*[\r\n]+|\s+(?!\S)|\s+")


def train(pre, decoder="bytelevel", normalizer=None, byte_fallback=False):
    from tokenizers import Tokenizer, decoders, models, trainers, pre_tokenizers
    tk = Tokenizer(models.BPE(byte_fallback=byte_fallback, unk_token=None if decoder == "bytelevel" else "<unk>"))
    tk.pre_tokenizer = pre
    if normalizer is not None:
        tk.normalizer = normalizer
    tk.decoder = decoders.ByteLevel() if decoder == "bytelevel" else decoders.Metaspace()
    alphabet = pre_tokenizers.ByteLevel.alphabet() if decoder == "bytelevel" else []
    tr = trainers.BpeTrainer(vocab_size=600, initial_alphabet=alphabet, show_progress=False,
                             special_tokens=["<unk>", "<s>", "</s>"])
    tk.train_from_iterator(TEXT * 20, tr)
    return tk


@unittest.skipUnless(CLI.exists(), "sml-cli.exe is not built")
class EngineCut(unittest.TestCase):
    def check(self, tk):
        try:
            import tokenizers  # noqa: F401
        except ImportError:
            self.skipTest("tokenizers is not installed")
        import sml_tok_check as T
        with tempfile.TemporaryDirectory() as d:
            tk.save(os.path.join(d, "tokenizer.json"))
            with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
                json.dump({"bos_token": "<s>", "eos_token": "</s>"}, f)
            ssf = T.vocab_only(d, os.path.join(d, "v.ssf"))
            lines = T.corpus()[:60] + TEXT[:8]
            got = T.engine(ssf, lines)
        want = [tk.encode(s, add_special_tokens=False).ids for s in lines]
        bad = [(s, w, g) for s, w, g in zip(lines, want, got) if w != g]
        self.assertFalse(bad, f"{len(bad)} lines differ, first: {bad[:1]}")

    def test_gpt2_byte_level(self):
        from tokenizers import pre_tokenizers
        self.check(train(pre_tokenizers.ByteLevel(add_prefix_space=False)))

    def test_llama3_regex_split(self):
        from tokenizers import Regex, pre_tokenizers
        self.check(train(pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(LLAMA3), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])))

    def test_digits_then_byte_level(self):
        from tokenizers import pre_tokenizers
        self.check(train(pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                  pre_tokenizers.ByteLevel(add_prefix_space=False)])))

    def test_punctuation_then_byte_level(self):
        from tokenizers import pre_tokenizers
        self.check(train(pre_tokenizers.Sequence([pre_tokenizers.Punctuation(behavior="contiguous"),
                                                  pre_tokenizers.ByteLevel(add_prefix_space=False),
                                                  pre_tokenizers.Digits(individual_digits=False)])))

    def test_script_classes_with_long_names(self):
        # Youtu-LLM's split names \p{Hangul}; the 4-byte class name in the file lost it (2026-10-20)
        from tokenizers import Regex, pre_tokenizers
        self.check(train(pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r"\p{Hangul}+|[一-龥぀-ゟ゠-ヿ]+|" + LLAMA3), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])))

    def test_metaspace(self):
        from tokenizers import pre_tokenizers
        self.check(train(pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first"),
                         decoder="metaspace", byte_fallback=False))


class Packing(unittest.TestCase):
    def test_classes_packed_for_the_steps_that_need_them(self):
        import sml_pretok as P
        pret = P.from_tokenizer_json({"model": {"type": "BPE", "vocab": {}, "merges": []},
                                      "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                                          {"type": "Punctuation", "behavior": "Contiguous"},
                                          {"type": "Digits", "individual_digits": True}]}})
        self.assertEqual([o["kind"] for o in pret["pre"]], [P.P_PUNCT, P.P_DIGITS])
        self.assertEqual(P._categories(pret["pre"]), ["N", "P", "s"])

    def test_long_class_names_become_short_ones(self):
        import sml_pretok as P
        got = P.short_names(r"\p{Hangul}+|\P{Han}|\p{L}")
        nm = got.split("}")[0].split("{")[1]
        self.assertLessEqual(len(nm), 4)
        self.assertEqual(P.ranges(nm), P.ranges("Hangul"))
        self.assertTrue(got.endswith(r"\P{Han}|\p{L}"))

    def test_unknown_step_refused_by_name(self):
        import sml_pretok as P
        with self.assertRaisesRegex(SystemExit, "WhitespaceSplit"):
            P.from_tokenizer_json({"model": {"type": "BPE"}, "pre_tokenizer": {"type": "WhitespaceSplit"}})


if __name__ == "__main__":
    unittest.main()
