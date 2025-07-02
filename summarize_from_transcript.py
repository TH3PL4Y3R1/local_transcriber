import os
import yaml
import requests

def summarize_text_lmstudio(text, config):
    """
    Summarize text using LM Studio local API.
    """
    api_url = config.get('lmstudio_api_url', 'http://localhost:1234/v1/chat/completions')
    llm_model = config.get('llm_model', None)
    max_tokens = config.get('max_tokens', 512)
    max_chunk = config.get('max_chunk', 2000)
    # Use sentence-based chunking if available
    try:
        from src.utils import sentence_chunk_text, filter_unrelated_sections
        chunks = sentence_chunk_text(text, max_chunk=max_chunk)
    except Exception:
        chunks = [text]
        filter_unrelated_sections = lambda x: x  # fallback: no filtering
    summary = ""
    for chunk in chunks:
        prompt = (
            "Resume el siguiente cuento infantil en español de forma clara y concisa, "
            "usando viñetas para los eventos principales. Ignora cualquier texto que no pertenezca al cuento.\n\n"
            f"Texto:\n{chunk}\n\nResumen:"
        )
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            response = requests.post(api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            chunk_summary = result['choices'][0]['message']['content'].strip()
            summary += chunk_summary + "\n"
        except Exception as e:
            summary += f"[LM STUDIO ERROR] {e}\n{chunk[:200]}...\n"
    try:
        summary = filter_unrelated_sections(summary.strip())
    except Exception:
        pass
    return summary

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
    summary = summarize_text_lmstudio(text, config)
    base = os.path.splitext(transcript_file)[0]
    summary_file = os.path.join(summary_dir, base + "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary written to {summary_file}")
