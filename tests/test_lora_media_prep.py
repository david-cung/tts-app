import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from apps import ui_utils


def _write_test_wav(path, seconds=1):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * (16_000 * seconds))


class LoraMediaPrepTests(unittest.TestCase):
    def test_ensure_sentence_punctuation(self):
        self.assertEqual(ui_utils._ensure_sentence_punctuation("xin chào"), "xin chào.")
        self.assertEqual(ui_utils._ensure_sentence_punctuation("đã xong."), "đã xong.")

    def test_persist_training_dataset_entries_creates_isolated_dataset(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clip_paths = []
            for index in range(3):
                clip_path = temp_path / f"clip_{index}.wav"
                _write_test_wav(clip_path)
                clip_paths.append(str(clip_path))

            dataset_dir = temp_path / "runs" / "giong_mai"
            saved = ui_utils.persist_training_dataset_entries(
                clip_paths,
                ["Câu một.", "Câu hai.", "Câu ba."],
                str(dataset_dir),
                clear_existing=True,
            )

            self.assertEqual(saved, 3)
            metadata = (dataset_dir / "metadata.csv").read_text(encoding="utf-8")
            self.assertEqual(metadata.count("|"), 3)
            self.assertTrue((dataset_dir / "raw_audio").is_dir())
            self.assertEqual(len(list((dataset_dir / "raw_audio").glob("*.wav"))), 3)

    def test_collect_training_clips_accepts_multiple_short_files(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            paths = []
            for index in range(3):
                clip_path = temp_path / f"sample_{index}.wav"
                _write_test_wav(clip_path, seconds=5)
                paths.append(str(clip_path))

            clips, warnings = ui_utils._collect_training_clips_from_files(
                paths,
                min_duration=3,
                max_duration=15,
                silence_ms=600,
            )

            self.assertEqual(len(clips), 3)
            self.assertEqual(warnings, [])
            for clip in clips:
                self.assertTrue(clip.endswith(".wav"))
                self.assertTrue(Path(clip).is_file())

    def test_collect_training_clips_requires_input(self):
        with self.assertRaises(ValueError):
            ui_utils._collect_training_clips_from_files([])
        with self.assertRaises(ValueError):
            ui_utils._collect_training_clips_from_files(None)


if __name__ == "__main__":
    unittest.main()
