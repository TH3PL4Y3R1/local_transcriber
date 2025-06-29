def summarize_text(text, config):
    # Use llama-cpp-python with Mistral 7B Instruct v0.3 GGUF model by default
    import os
    llm_model = config.get('llm_model', 'mistral-7b-instruct-v0.3.Q4_K_M.gguf')
    if llm_model.endswith('.gguf'):
        try:
            from llama_cpp import Llama
            model_path = config.get('models_dir', 'models')
            model_file = os.path.join(model_path, llm_model)
            llm = Llama(model_path=model_file, n_ctx=4096)
            prompt = f"Resumí el siguiente texto en español de forma concisa y clara, usando viñetas si es posible.\n\nTexto:\n{text[:3500]}\n\nResumen:"
            output = llm(prompt, max_tokens=256, stop=["\n\n"])
            return output['choices'][0]['text'].strip()
        except Exception as e:
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
