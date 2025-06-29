import os
import yaml
from src.pipeline import run_pipeline

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    input_dir = config.get('input_dir', 'audio_input')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.m4a')):
            audio_path = os.path.join(input_dir, filename)
            transcript, summary = run_pipeline(audio_path, config)
            base = os.path.splitext(filename)[0]
            with open(os.path.join(config['transcripts_dir'], base + ".txt"), "w", encoding="utf-8") as tf:
                tf.write(transcript)
            with open(os.path.join(config['summaries_dir'], base + "_summary.txt"), "w", encoding="utf-8") as sf:
                sf.write(summary)
