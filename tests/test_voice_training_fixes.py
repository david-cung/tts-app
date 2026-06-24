import unittest
from pathlib import Path

import torch

from finetune.data_scripts.filter_data import text_filter
from finetune.train import apply_small_dataset_training_config, preprocess_sample
from finetune.voice_training_pipeline import _recommended_max_steps


class _FakeTokenizer:
    """Minimal tokenizer to validate preprocess_sample structure."""

    SPECIALS = {
        "<|TEXT_PROMPT_START|>": 1,
        "<|TEXT_PROMPT_END|>": 2,
        "<|SPEECH_GENERATION_START|>": 3,
        "<|SPEECH_GENERATION_END|>": 4,
        "<|emotion_0|>": 5,
    }
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.SPECIALS[token]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for token in text.replace("<|", " <|").replace("|>", "|> ").split():
            if token in self.SPECIALS:
                ids.append(self.SPECIALS[token])
            elif token.startswith("<|speech_"):
                ids.append(1000 + int(token[len("<|speech_"):-2]))
            else:
                ids.append(500)
        return ids


class VoiceTrainingFixesTests(unittest.TestCase):
    def test_encode_data_uses_distill_codec(self):
        source = Path("finetune/data_scripts/encode_data.py").read_text(encoding="utf-8")
        self.assertIn("DistillNeuCodec", source)
        self.assertNotIn("NeuCodec", source.replace("DistillNeuCodec", ""))

    def test_tiny_dataset_caps_steps_and_lr(self):
        tuned = apply_small_dataset_training_config(
            {"max_steps": 800, "learning_rate": 1e-4},
            sample_count=4,
        )
        self.assertLessEqual(tuned["max_steps"], 250)
        self.assertLessEqual(tuned["learning_rate"], 3e-5)

    def test_larger_dataset_gets_more_steps(self):
        tuned = apply_small_dataset_training_config(
            {"max_steps": 800, "learning_rate": 1e-4},
            sample_count=20,
        )
        self.assertGreaterEqual(tuned["max_steps"], 500)
        self.assertGreaterEqual(tuned["learning_rate"], 8e-5)

    def test_build_lora_config_scales_with_dataset(self):
        from finetune.configs.lora_config import build_lora_config

        small = build_lora_config(5)
        medium = build_lora_config(15)
        large = build_lora_config(40)
        self.assertEqual(small.r, 8)
        self.assertEqual(medium.r, 16)
        self.assertEqual(large.r, 32)
        # alpha should track rank for stable scaling
        self.assertEqual(medium.lora_alpha, 32)
        self.assertEqual(large.lora_alpha, 64)

    def test_text_filter_allows_digits(self):
        self.assertTrue(text_filter("Đội bóng thắng 3-2 năm 2026."))
        self.assertTrue(text_filter("Có 10 người tham gia."))

    def test_text_filter_rejects_missing_punctuation(self):
        self.assertFalse(text_filter("xin chào"))

    def test_recommended_max_steps_scales_with_dataset(self):
        # Recommended steps roughly scale with sample count, with sensible floors.
        self.assertGreaterEqual(_recommended_max_steps(3), 150)
        self.assertGreaterEqual(_recommended_max_steps(8), 400)
        self.assertGreaterEqual(_recommended_max_steps(20), 500)
        self.assertEqual(_recommended_max_steps(5, requested=250), 250)

    def test_preprocess_sample_supervises_target_codes_only(self):
        tok = _FakeTokenizer()
        out = preprocess_sample(
            {"phones": "tgt", "codes": [11, 12]},
            tok,
            max_len=32,
            emotion_tag="<|emotion_0|>",
        )
        ids = out["input_ids"].tolist()
        labels = out["labels"].tolist()

        # prefix: TPS, emotion, target phones, TPE, SGS
        self.assertEqual(ids[0], tok.SPECIALS["<|TEXT_PROMPT_START|>"])
        self.assertEqual(ids[1], tok.SPECIALS["<|emotion_0|>"])
        self.assertIn(tok.SPECIALS["<|SPEECH_GENERATION_START|>"], ids)
        self.assertIn(1011, ids)  # target code 11
        self.assertIn(1012, ids)  # target code 12

        # Only target codes + speech-end are supervised; prefix masked.
        supervised = [i for i, lbl in zip(ids, labels) if lbl != -100]
        self.assertEqual(
            supervised,
            [1011, 1012, tok.SPECIALS["<|SPEECH_GENERATION_END|>"]],
        )
        for token_id, lbl in zip(ids, labels):
            if token_id == tok.pad_token_id:
                self.assertEqual(lbl, -100)

    def test_preprocess_sample_never_fully_masked_when_prefix_overflows(self):
        tok = _FakeTokenizer()
        # Huge phoneme prefix that would otherwise crowd out all target labels.
        out = preprocess_sample(
            {"phones": " ".join(["w"] * 50), "codes": [11, 12, 13]},
            tok,
            max_len=16,
            emotion_tag="<|emotion_0|>",
        )
        labels = out["labels"].tolist()
        supervised = [lbl for lbl in labels if lbl != -100]
        self.assertGreaterEqual(len(supervised), 1)
        self.assertEqual(len(out["input_ids"]), 16)


if __name__ == "__main__":
    unittest.main()
