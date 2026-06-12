# Docker Web UI Image

This image packages the VieNeu-TTS Gradio Web UI for CPU testing. It runs the
default v3 Turbo ONNX path and downloads model files into a Docker volume on the
first model load.

## Build

```bash
docker build -f docker/Dockerfile.web -t vieneu-tts-web:cpu .
```

On Apple Silicon, that command creates a `linux/arm64` image. Build a
`linux/amd64` image if you want to send it to most Windows or Linux testers:

```bash
docker buildx build --platform linux/amd64 -f docker/Dockerfile.web -t vieneu-tts-web:cpu-amd64 --load .
```

Cross-building `linux/amd64` from Apple Silicon can be very slow because
`sea-g2p` currently has no Linux x86_64 wheel and must compile from source under
QEMU. If possible, build this image on an amd64 machine or CI runner.

## Run

```bash
docker run --rm \
  -p 7860:7860 \
  -v vieneu_hf_cache:/data/huggingface \
  -v vieneu_outputs:/app/outputs \
  vieneu-tts-web:cpu
```

Open:

```text
http://127.0.0.1:7860
```

The first click on **Tải Model** needs internet access to download the v3 Turbo
model from Hugging Face. Later runs reuse the `vieneu_hf_cache` Docker volume.

## Run With Compose

```bash
docker compose -f docker/docker-compose.web.yml up --build
```

## Export To Send Elsewhere

For the current machine architecture:

```bash
docker save vieneu-tts-web:cpu | gzip > vieneu-tts-web-cpu.tar.gz
```

For `linux/amd64`:

```bash
docker save vieneu-tts-web:cpu-amd64 | gzip > vieneu-tts-web-cpu-amd64.tar.gz
```

The tester can load and run it with:

```bash
gunzip -c vieneu-tts-web-cpu.tar.gz | docker load
docker run --rm -p 7860:7860 -v vieneu_hf_cache:/data/huggingface vieneu-tts-web:cpu
```

Or, for the amd64 image:

```bash
gunzip -c vieneu-tts-web-cpu-amd64.tar.gz | docker load
docker run --rm -p 7860:7860 -v vieneu_hf_cache:/data/huggingface vieneu-tts-web:cpu-amd64
```

If the tester has no internet access, you need a larger offline image or a
pre-populated Hugging Face cache volume containing the model files.
