import os
from tqdm import tqdm
from .preprocess import preprocess_audio
from .transcribe import transcribe_audio
from .summarize import summarize_text

def run_pipeline(audio_path, config):
    # Preprocess audio (chunking, conversion)
    chunk_paths = preprocess_audio(audio_path, config)
    transcripts = []
    # Progress bar for transcription
    for chunk in tqdm(chunk_paths, desc=f"Transcribing {os.path.basename(audio_path)}", unit="chunk"):
        text = transcribe_audio(chunk, config)
        transcripts.append(text)
    full_transcript = '\n'.join(transcripts)
    summary = summarize_text(full_transcript, config)
    return full_transcript, summary
