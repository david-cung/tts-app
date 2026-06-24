import os
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

from finetune.train import _is_lora_checkpoint_dir, _resolve_lora_checkpoint_dir


class TrainCheckpointTests(unittest.TestCase):
    def test_resolve_latest_checkpoint(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "football")
            checkpoint = os.path.join(run_dir, "checkpoint-500")
            os.makedirs(checkpoint)
            with open(os.path.join(checkpoint, "adapter_config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")

            resolved = _resolve_lora_checkpoint_dir(temp_dir, "football")
            self.assertTrue(resolved.endswith("checkpoint-500"))
            self.assertTrue(_is_lora_checkpoint_dir(resolved))

    def test_resolve_run_root_adapter(self):
        with TemporaryDirectory() as temp_dir:
            run_dir = os.path.join(temp_dir, "football")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "adapter_config.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")

            resolved = _resolve_lora_checkpoint_dir(temp_dir, "football")
            self.assertEqual(resolved, run_dir)


class TrainDeviceConfigTests(unittest.TestCase):
    def test_mps_config_reduces_memory(self):
        with mock.patch("finetune.train._use_mps_training", return_value=True):
            from finetune.train import apply_device_training_config

            tuned = apply_device_training_config(
                {
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_seq_len": 2048,
                }
            )

        self.assertEqual(tuned["per_device_train_batch_size"], 1)
        self.assertGreaterEqual(tuned["gradient_accumulation_steps"], 4)
        self.assertEqual(tuned["max_seq_len"], 1024)
        self.assertTrue(tuned["gradient_checkpointing"])


if __name__ == "__main__":
    unittest.main()
