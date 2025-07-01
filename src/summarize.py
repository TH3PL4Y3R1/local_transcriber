def summarize_text(text, config):
    # Summarize text using llama-cpp-python with GGUF model, auto-detect if not found
    import os
    llm_model = config.get('llm_model', None)
    model_path = config.get('models_dir', 'models')
    if not llm_model or not os.path.exists(os.path.join(model_path, llm_model)):
        available = [f for f in os.listdir(model_path) if f.lower().endswith('.gguf')]
        if not available:
            return '[LLAMA ERROR] No GGUF models found in models directory.'
        msg = '[LLAMA ERROR] Model not found. Available models:\n'
        for idx, fname in enumerate(available):
            msg += f'  [{idx+1}] {fname}\n'
        msg += 'Please update config.yaml with one of the above filenames under llm_model.'
        return msg
    if llm_model.endswith('.gguf'):
        try:
            from llama_cpp import Llama
            from src.utils import sentence_chunk_text, filter_unrelated_sections
            from tqdm import tqdm
            model_file = os.path.join(model_path, llm_model)
            llm = Llama(model_path=model_file, n_ctx=4096, n_gpu_layers=48)
            max_tokens = config.get('max_tokens', 512)
            max_chunk = config.get('max_chunk', 2000)
            # Use sentence-based chunking for better coherence
            chunks = sentence_chunk_text(text, max_chunk=max_chunk)
            summary = ""
            for chunk in tqdm(chunks, desc="Summarizing text", unit="chunk"):
                prompt = (
                    "Resume el siguiente cuento infantil en español de forma clara y concisa, "
                    "usando viñetas para los eventos principales. Ignora cualquier texto que no pertenezca al cuento.\n\n"
                    f"Texto:\n{chunk}\n\nResumen:"
                )
                output = llm(prompt, max_tokens=max_tokens)
                result = output['choices'][0]['text'].strip()
                if not result:
                    continue
                summary += result + "\n"
            # Post-process to remove unrelated/repeated sections
            summary = filter_unrelated_sections(summary.strip())
            return summary
        except Exception as e:
            return f"[LLAMA ERROR] {e}\n{text[:200]}..."
    else:
        try:
            from transformers import pipeline
            from src.utils import sentence_chunk_text, filter_unrelated_sections
            from tqdm import tqdm
            summarizer = pipeline("summarization", model=llm_model)
            max_chunk = 1000
            chunks = sentence_chunk_text(text, max_chunk=max_chunk)
            summary = ""
            for chunk in tqdm(chunks, desc="Summarizing text", unit="chunk"):
                out = summarizer(chunk, max_length=256, min_length=40, do_sample=False)
                summary += out[0]['summary_text'] + "\n"
            summary = filter_unrelated_sections(summary.strip())
            return summary
        except Exception as e:
            return f"[TRANSFORMERS ERROR] {e}\n{text[:200]}..."
