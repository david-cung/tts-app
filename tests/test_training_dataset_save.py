import unittest
import wave
from unittest import mock

from apps import ui_utils


def _write_test_wav(path):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 16_000)


class SaveTrainingDatasetTests(unittest.TestCase):
    def test_reports_success(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            app_dir = temp_path / "apps"
            app_dir.mkdir()
            audio_path = temp_path / "sample.wav"
            _write_test_wav(audio_path)

            notifications = []
            with (
                mock.patch.object(ui_utils, "__file__", str(app_dir / "ui_utils.py")),
                mock.patch.object(
                    ui_utils.gr,
                    "Info",
                    side_effect=lambda message, **kwargs: notifications.append((message, kwargs)),
                ),
            ):
                status, rows = ui_utils.save_voice_training_dataset(
                    [str(audio_path)],
                    script_text="Xin chào",
                )

            self.assertIn("Lưu dataset thành công", status)
            self.assertIn("1 mẫu mới", status)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                notifications[0][0],
                "Đã lưu 1 mẫu mới. Dataset hiện có 1 mẫu.",
            )
            self.assertEqual(
                (temp_path / "finetune" / "dataset" / "metadata.csv").read_text(
                    encoding="utf-8"
                ),
                "sample.wav|Xin chào\n",
            )

    def test_reports_error(self):
        notifications = []
        with mock.patch.object(
            ui_utils.gr,
            "Warning",
            side_effect=lambda message, **kwargs: notifications.append((message, kwargs)),
        ):
            status, rows = ui_utils.save_voice_training_dataset([], script_text="")

        self.assertIn("Không thể lưu dataset", status)
        self.assertEqual(rows, [])
        self.assertTrue(notifications)


if __name__ == "__main__":
    unittest.main()
