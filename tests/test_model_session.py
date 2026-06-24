import unittest
from tempfile import TemporaryDirectory

from apps.model_session import load_model_session, save_model_session, should_auto_load_model


class ModelSessionTests(unittest.TestCase):
    def test_save_and_load_session(self):
        with TemporaryDirectory() as temp_dir:
            path = save_model_session(
                temp_dir,
                {
                    "backbone": "VieNeu-TTS-v3-Turbo (Thử nghiệm)",
                    "codec": "VieNeu-Codec",
                    "device": "Auto",
                    "force_lmdeploy": False,
                },
            )
            self.assertTrue(path.endswith("model_session.json"))
            loaded = load_model_session(temp_dir)
            self.assertEqual(loaded["backbone"], "VieNeu-TTS-v3-Turbo (Thử nghiệm)")

    def test_should_auto_load_when_session_exists(self):
        with TemporaryDirectory() as temp_dir:
            self.assertFalse(should_auto_load_model(temp_dir))
            save_model_session(temp_dir, {"backbone": "test"})
            self.assertTrue(should_auto_load_model(temp_dir))


if __name__ == "__main__":
    unittest.main()
