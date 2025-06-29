from pydub import AudioSegment
import os

def preprocess_audio(audio_path, config):
    chunk_size = config.get('chunk_size', 300) * 1000  # ms
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        chunk_path = os.path.join(config['chunks_dir'], f"chunk_{i//chunk_size}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks
