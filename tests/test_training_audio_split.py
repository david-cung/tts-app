from apps.ui_utils import _build_vad_training_segments


def test_build_vad_segments_combines_nearby_speech():
    segments = _build_vad_training_segments(
        [{"start": 0, "end": 32_000}, {"start": 36_000, "end": 80_000}],
        total_samples=160_000,
        min_samples=48_000,
        max_samples=240_000,
    )

    assert segments == [(0, 80_000)]


def test_build_vad_segments_splits_long_region():
    segments = _build_vad_training_segments(
        [{"start": 0, "end": 500_000}],
        total_samples=500_000,
        min_samples=48_000,
        max_samples=240_000,
    )

    assert segments == [(0, 166_667), (166_667, 333_334), (333_334, 500_000)]


def test_build_vad_segments_pads_short_isolated_region():
    segments = _build_vad_training_segments(
        [{"start": 40_000, "end": 56_000}],
        total_samples=160_000,
        min_samples=48_000,
        max_samples=240_000,
    )

    assert segments == [(24_000, 72_000)]
