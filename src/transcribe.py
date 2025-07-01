import whisper

def transcribe_audio(audio_path, config):
    model = whisper.load_model(config.get('whisper_model', 'base'), device='cuda')
    language = config.get('whisper_language', 'es')
    result = model.transcribe(audio_path, language=language)
    return result['text']
