import os
import yaml
from src.summarize import summarize_text

if __name__ == "__main__":
    # Ask user for transcript filename
    transcript_dir = "transcripts"
    summary_dir = "summaries"
    transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith(".txt")]
    if not transcript_files:
        print("No transcript files found in 'transcripts/'.")
        exit(1)
    print("Available transcript files:")
    for idx, fname in enumerate(transcript_files):
        print(f"  [{idx+1}] {fname}")
    choice = input("Enter the number of the transcript to summarize: ")
    try:
        idx = int(choice) - 1
        transcript_file = transcript_files[idx]
    except Exception:
        print("Invalid selection.")
        exit(1)
    with open(os.path.join(transcript_dir, transcript_file), "r", encoding="utf-8") as f:
        text = f.read()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    summary = summarize_text(text, config)
    base = os.path.splitext(transcript_file)[0]
    summary_file = os.path.join(summary_dir, base + "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary written to {summary_file}")
