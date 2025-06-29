def summarize_text(text, config):
    # Use llama-cpp-python with GGUF model, auto-detect if not found
    import os
    llm_model = config.get('llm_model', None)
    model_path = config.get('models_dir', 'models')
    if not llm_model or not os.path.exists(os.path.join(model_path, llm_model)):
        # List available GGUF models
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
            model_file = os.path.join(model_path, llm_model)
            # Use GPU for inference: set n_gpu_layers to use most of your 8GB VRAM (try 40-50 for 7B model)
            llm = Llama(model_path=model_file, n_ctx=4096, n_gpu_layers=48)
            prompt = f"Resumí el siguiente texto en español de forma concisa y clara, usando viñetas si es posible.\n\nTexto:\n{text[:2000]}\n\nResumen:"
            output = llm(prompt, max_tokens=256)
            result = output['choices'][0]['text'].strip()
            # --- NOTE FOR FUTURE DEBUGGING ---
            # If you are still seeing CPU usage, the cuBLAS (GPU) build is not being loaded.
            # Check your llama-cpp-python install and logs for CUDA or GPU assignment.
            # See https://github.com/abetlen/llama-cpp-python/discussions/1587 for troubleshooting.
            if not result:
                print('[DEBUG] LLM returned empty summary. Full output:', output)
                print('[NOTE] Check if llama-cpp-python is using CPU or GPU. See code comments for help.')
                return '[LLAMA ERROR] LLM returned empty summary.'
            print('[DEBUG] LLM summary:', result)
            return result
        except Exception as e:
            print(f'[DEBUG] Exception in LLM summarization: {e}')
            print('[NOTE] Check if llama-cpp-python is using CPU or GPU. See code comments for help.')
            return f"[LLAMA ERROR] {e}\n{text[:200]}..."
    else:
        try:
            from transformers import pipeline
            summarizer = pipeline("summarization", model=llm_model)
            max_chunk = 1000
            chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
            summary = ""
            for chunk in chunks:
                out = summarizer(chunk, max_length=256, min_length=40, do_sample=False)
                summary += out[0]['summary_text'] + "\n"
            return summary.strip()
        except Exception as e:
            return f"[TRANSFORMERS ERROR] {e}\n{text[:200]}..."
