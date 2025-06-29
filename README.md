# Local AI Transcriber & Summarizer

A fully local, GPU-accelerated tool to transcribe and summarize audio recordings (lectures, talks, etc.) using Whisper and local LLMs. No cloud required.

## Features
- Audio transcription (Whisper, local, GPU-accelerated if supported)
- Audio chunking and preprocessing (pydub/ffmpeg)
- Text summarization (LLaMA 2, Mistral, etc. via transformers or llama-cpp-python)
- All processing is local and private
- Batch processing of multiple audio files
- Configurable chunk size, model paths, and output directories

## Project Structure
```
local_transcriber/
├── audio_input/      # Place raw audio files here
├── audio_chunks/     # Stores processed audio chunks
├── transcripts/      # Stores raw and cleaned transcriptions
├── summaries/        # Stores generated summaries
├── models/           # Downloaded or converted LLM/Whisper models
├── src/              # Main source code
├── config.yaml       # Configurable parameters
├── requirements.txt  # Python dependencies
├── main.py           # CLI entry point
├── .gitignore        # Git ignore rules
└── README.md         # Project documentation
```

## Setup
1. **Install Python 3.10** and create a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```
2. **Install FFmpeg** and add it to your PATH (see project summary for details).
3. (Optional) Download and place LLM/Whisper models in the `models/` folder for offline use.

## Usage
1. Place audio files in `audio_input/` (supported: .wav, .mp3, .m4a)
2. Run:
   ```sh
   python main.py
   ```
3. Transcripts will be saved in `transcripts/`, summaries in `summaries/`.

## Configuration
Edit `config.yaml` to set chunk size, model names/paths, and directory locations.

## Notes
- The `.gitignore` excludes all generated data, models, and virtual environment files.
- Summarization is a placeholder; implement your preferred LLM logic in `src/summarize.py`.
- For GPU acceleration, ensure you have the correct CUDA-enabled PyTorch installed and a compatible GPU.

---

For more details, see `summary.txt` or ask for help!
