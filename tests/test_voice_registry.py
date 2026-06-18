import unittest
from unittest import mock

from finetune import voice_registry
from finetune.voice_training_pipeline import validate_dataset_ready, validate_training_environment


class VoiceRegistryTests(unittest.TestCase):
    def test_register_and_list_voice(self):
        with mock.patch.object(voice_registry, "registry_path") as registry_path_mock:
            with mock.patch.object(voice_registry, "registry_dir") as registry_dir_mock:
                registry_dir_mock.return_value = "/tmp/test_user_voices"
                registry_path_mock.return_value = "/tmp/test_user_voices/registry.json"

                with (
                    mock.patch("finetune.voice_registry.os.path.isdir", return_value=True),
                    mock.patch("finetune.voice_registry.os.makedirs"),
                    mock.patch("finetune.voice_registry.save_registry") as save_registry,
                    mock.patch("finetune.voice_registry.load_registry", return_value={"voices": {}}),
                ):
                    entry = voice_registry.register_voice(
                        display_name="Giọng Mai",
                        lora_path="/tmp/finetune/output/giong_mai",
                        voice_preset_id="giong_mai",
                    )

                self.assertEqual(entry["display_name"], "Giọng Mai")
                save_registry.assert_called_once()

    def test_dropdown_choice_format(self):
        with mock.patch.object(
            voice_registry,
            "list_registered_voices",
            return_value=[{"voice_id": "giong_mai", "display_name": "Giọng Mai"}],
        ):
            choices = voice_registry.list_user_voice_dropdown_choices()

        self.assertEqual(choices, [("🎤 Giọng Mai", "user_voice:giong_mai")])


class VoiceTrainingPipelineTests(unittest.TestCase):
    def test_validate_environment_without_torch(self):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            ok, message = validate_training_environment()

        self.assertFalse(ok)
        self.assertIn("uv sync --group gpu", message)

    def test_validate_dataset_requires_samples(self):
        with mock.patch(
            "finetune.voice_training_pipeline._count_metadata_lines",
            return_value=1,
        ):
            with self.assertRaises(ValueError):
                validate_dataset_ready(min_samples=3)


if __name__ == "__main__":
    unittest.main()
