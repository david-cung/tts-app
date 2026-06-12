# TTS App

A local web application for Vietnamese text-to-speech, reference-based voice cloning, multi-speaker generation, and voice-training dataset preparation.

The application can run with Docker or directly in Python. Docker is recommended because it packages FFmpeg, Faster-Whisper, and the required runtime dependencies.

## Features

- Text-to-speech generation through a Gradio web interface.
- Built-in preset voices.
- Reference-based voice cloning from a short audio sample.
- Multi-speaker conversation generation.
- Multi-file WAV upload for dataset preparation.
- Automatic segmentation of long audio or video files with Silero VAD.
- Configurable minimum duration, maximum duration, and silence threshold for segmentation.
- Automatic Vietnamese transcription with Faster-Whisper.
- WAV-to-transcript preview and validation before saving.
- Persistent training datasets through a host bind mount.
- CPU inference with ONNX Runtime and optional GPU dependencies.

## Requirements

### Docker

- Docker Desktop.
- Internet access during the first image build and initial model downloads.
- At least 8 GB of system RAM. Additional free disk space is recommended for images and model caches.

### Local Python

- Python 3.12.
- `uv`.
- FFmpeg when processing video or non-WAV audio locally.

## Run with Docker

Build and start the application:

```powershell
docker compose -f docker/docker-compose.web.yml up --build -d
```

Open the web interface:

```text
http://127.0.0.1:7860
```

Follow the logs:

```powershell
docker compose -f docker/docker-compose.web.yml logs -f web
```

Stop the application:

```powershell
docker compose -f docker/docker-compose.web.yml down
```

The Docker image includes:

- FFmpeg for audio and video decoding.
- Faster-Whisper for transcription.
- Silero VAD for speech activity detection.
- ONNX Runtime for CPU inference.

Models downloaded from Hugging Face are stored in a Docker volume and remain available across container restarts.

## Run Locally

Install `uv` on Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install the CPU dependencies:

```powershell
uv sync
```

Install the optional transcription dependencies:

```powershell
uv sync --extra asr
```

Start the web interface:

```powershell
uv run python -m apps.gradio_main
```

For an NVIDIA GPU environment:

```powershell
uv sync --group gpu
uv run python -m apps.gradio_main
```

## Text-to-Speech

1. Open the web interface.
2. Select a model, device, and codec.
3. Click **Load Model**.
4. Open the **Preset** tab.
5. Enter the text to synthesize.
6. Click **Start**.

The initial model download may take several minutes depending on the network connection.

## Voice Cloning

1. Open the **Voice Cloning** tab.
2. Upload a reference recording containing one speaker.
3. Use a clean sample approximately 3-10 seconds long.
4. Enter the new text to synthesize.
5. Click **Start**.

The reference recording should contain no background music, minimal reverberation, and a consistent speaking volume. Voice cloning does not retrain the model; the recording is used only as a reference during generation.

Only clone a voice when you have the speaker's permission or otherwise have the legal right to use it.

## Split Long Audio or Video

Open the **Voice Training** tab:

1. Upload a file through **Long audio/video to split**.
2. Set the minimum clip duration. The default is `3 seconds`.
3. Set the maximum clip duration. The default is `15 seconds`.
4. Set the silence threshold. The default is `600 ms`.
5. Click **Split into WAV files**.

The generated clips are added automatically to the **Training WAV files** list.

Supported formats:

```text
WAV, MP3, FLAC, M4A, OGG, AAC, MP4, MOV, MKV, WEBM
```

Silero VAD detects speech regions and prefers natural silence boundaries. Continuous speech longer than the configured maximum duration is divided into balanced clips.

## Generate Transcripts and Save a Dataset

After uploading WAV files or splitting a long recording:

1. Select a Whisper model.
2. Click **Generate Transcript**.
3. Review and correct every transcript so it matches the recording exactly.
4. Click **Preview Mapping**.
5. Verify the WAV and transcript order.
6. Click **Save Dataset**.

Available Whisper models:

| Model | Characteristics |
| --- | --- |
| `tiny` | Fastest, lower transcription accuracy |
| `base` | Lightweight and suitable for quick tests |
| `small` | Recommended balance of speed and accuracy |
| `medium` | Higher accuracy with greater CPU and memory usage |

Faster-Whisper downloads the selected model on first use and stores it in the model cache.

The dataset is saved with the following structure:

```text
finetune/dataset/
|-- metadata.csv
`-- raw_audio/
    |-- audio_001.wav
    `-- audio_002.wav
```

The `metadata.csv` format is:

```text
audio_001.wav|The exact transcript for the first recording.
audio_002.wav|The exact transcript for the second recording.
```

Docker bind-mounts the dataset directory directly into the container:

```text
C:\code\tts-app\finetune\dataset
    -> /app/finetune/dataset
```

The dataset remains on the host after the container is removed.

## Dataset Guidelines

Each WAV file should:

- Be approximately 3-15 seconds long.
- Contain only one speaker.
- Contain no background music or sound effects.
- Use consistent volume with minimal reverberation.
- Have a transcript that exactly matches the spoken content.

Suggested dataset sizes:

| Purpose | Total duration |
| --- | --- |
| Pipeline validation | 15-30 minutes |
| Experimental fine-tuning | Approximately 1 hour |
| More stable fine-tuning | 2-4 hours |

A few recordings are not enough to fine-tune a stable voice. Use reference-based voice cloning when only a small amount of audio is available.

## LoRA Fine-Tuning

The web interface prepares the dataset but does not run training. Fine-tuning requires a GPU environment and the scripts under `finetune/`.

Install the GPU dependencies:

```powershell
uv sync --group gpu
```

Filter invalid or unsuitable samples:

```powershell
uv run python finetune/data_scripts/filter_data.py
```

Encode the audio samples:

```powershell
uv run python finetune/data_scripts/encode_data.py
```

Start training:

```powershell
uv run python finetune/train.py
```

Training output is written under:

```text
finetune/output/<run-name>
```

An NVIDIA GPU with at least 12 GB of VRAM is recommended. Review the training configuration before starting:

```text
finetune/configs/lora_config.py
```

## Data and Cache Locations

| Data | Location |
| --- | --- |
| Training dataset | `finetune/dataset` |
| Fine-tuning output | `finetune/output` |
| Generated Docker output | Docker volume managed by Compose |
| Docker model cache | Docker volume managed by Compose |

Do not commit private recordings, datasets, or model files without verifying that you have the necessary data rights.

## Troubleshooting

### Port 7860 Is Already in Use

Run the container on another host port:

```powershell
$env:PORT=7861
docker compose -f docker/docker-compose.web.yml up -d
```

Open `http://127.0.0.1:7861`.

### Transcript Generation Fails

- Rebuild the image from the latest `docker/Dockerfile.web`.
- Confirm internet access during the initial Whisper model download.
- Inspect logs with `docker compose -f docker/docker-compose.web.yml logs -f web`.

### Video Segmentation Fails

- Confirm that the file format is supported.
- Confirm that the file contains an audio track.
- Docker installations already include FFmpeg in the web image.

### The Dataset Does Not Appear on the Host

Review the `web` service in `docker/docker-compose.web.yml`. The bind mount destination must be:

```text
/app/finetune/dataset
```

## Project Structure

```text
apps/                       Gradio interface and application workflows
src/                        Text-to-speech backend
docker/                     Dockerfiles and Compose configuration
finetune/                   Dataset preparation and fine-tuning scripts
tests/                      Automated tests
examples/                   Example code and audio files
config.yaml                 Model and text chunking configuration
pyproject.toml              Dependencies and application entry points
```

## License

See [LICENSE](LICENSE).
