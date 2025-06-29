import whisper

def transcribe_audio(audio_path, config):
    model = whisper.load_model(config.get('whisper_model', 'base'))
    result = model.transcribe(audio_path)
    return result['text']
