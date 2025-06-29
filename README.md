# Local AI Transcriber & Summarizer

A fully local, GPU-accelerated tool to transcribe and summarize audio recordings (lectures, talks, etc.) using Whisper and local LLMs. No cloud required.

## Features
- Audio transcription (Whisper, local)
- Audio chunking and preprocessing (pydub/ffmpeg)
- Text summarization (LLaMA 2, Mistral, etc.)
- All processing is local and private

## Usage
1. Place audio files in `audio_input/`
2. Run `main.py` to process and summarize
3. Results appear in `transcripts/` and `summaries/`

See `config.yaml` for configuration options.
