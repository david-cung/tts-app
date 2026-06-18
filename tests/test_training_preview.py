import json
import os
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from apps import ui_utils
from finetune import preview_trained_voice as preview_mod


def _write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 16_000)


class TrainingPreviewTests(unittest.TestCase):
    def test_list_training_checkpoints_finds_lora_dirs(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint = temp_path / "finetune" / "output" / "demo-lora"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(preview_mod, "_project_root", return_value=str(temp_path)):
                checkpoints = preview_mod.list_training_checkpoints()

            self.assertEqual(checkpoints, ["finetune/output/demo-lora"])

    def test_find_dataset_reference_reads_metadata(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_dir = temp_path / "finetune" / "dataset"
            raw_audio = dataset_dir / "raw_audio"
            raw_audio.mkdir(parents=True)
            wav_path = raw_audio / "clip.wav"
            _write_test_wav(wav_path)
            (dataset_dir / "metadata.csv").write_text(
                "clip.wav|Xin chào dataset\n",
                encoding="utf-8",
            )

            ref_audio, ref_text = preview_mod.find_dataset_reference(str(dataset_dir))
            self.assertEqual(ref_audio, str(wav_path))
            self.assertEqual(ref_text, "Xin chào dataset")

    def test_on_training_checkpoint_selected_loads_preview_sample(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint = temp_path / "finetune" / "output" / "demo-lora"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            sample_path = checkpoint / preview_mod.PREVIEW_SAMPLE_NAME
            _write_test_wav(sample_path)

            rel_checkpoint = "finetune/output/demo-lora"
            with mock.patch.object(preview_mod, "_project_root", return_value=str(temp_path)):
                audio_path, status = ui_utils.on_training_checkpoint_selected(rel_checkpoint)

            self.assertEqual(audio_path, str(sample_path))
            self.assertIn("preview_sample.wav", status)

    def test_generate_training_preview_writes_meta(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint = temp_path / "finetune" / "output" / "demo-lora"
            raw_audio = temp_path / "finetune" / "dataset" / "raw_audio"
            raw_audio.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
            ref_audio = raw_audio / "clip.wav"
            _write_test_wav(ref_audio)
            (temp_path / "finetune" / "dataset" / "metadata.csv").write_text(
                "clip.wav|Xin chào\n",
                encoding="utf-8",
            )

            fake_output = checkpoint / preview_mod.PREVIEW_SAMPLE_NAME
            fake_output.write_bytes(b"RIFF")

            class FakeTTS:
                def load_lora_adapter(self, *_args, **_kwargs):
                    return True

                def infer(self, *_args, **_kwargs):
                    return [0.0, 0.1, 0.0]

                def save(self, _audio, path):
                    Path(path).write_bytes(b"RIFF")

                def close(self):
                    return None

            rel_checkpoint = "finetune/output/demo-lora"
            with (
                mock.patch.object(preview_mod, "_project_root", return_value=str(temp_path)),
                mock.patch.dict(
                    "sys.modules",
                    {
                        "torch": mock.MagicMock(
                            cuda=mock.MagicMock(is_available=mock.Mock(return_value=False)),
                            backends=mock.MagicMock(
                                mps=mock.MagicMock(is_available=mock.Mock(return_value=False))
                            ),
                        )
                    },
                ),
                mock.patch("vieneu.standard.VieNeuTTS", return_value=FakeTTS()),
            ):
                output_path = preview_mod.generate_training_preview(
                    rel_checkpoint,
                    text=preview_mod.DEFAULT_PREVIEW_TEXT,
                )

            self.assertTrue(os.path.isfile(output_path))
            meta = json.loads((checkpoint / preview_mod.PREVIEW_META_NAME).read_text(encoding="utf-8"))
            self.assertEqual(meta["text"], preview_mod.DEFAULT_PREVIEW_TEXT)


if __name__ == "__main__":
    unittest.main()
