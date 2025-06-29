def summarize_text(text, config):
    # Try to use transformers first, fallback to llama-cpp-python if specified in config
    llm_model = config.get('llm_model', 'llama-2-7b.gguf')
    if llm_model.endswith('.gguf'):
        # Use llama-cpp-python
        try:
            from llama_cpp import Llama
            model_path = config.get('models_dir', 'models')
            model_file = os.path.join(model_path, llm_model)
            llm = Llama(model_path=model_file, n_ctx=2048)
            prompt = f"Summarize the following text in Spanish:\n{text[:2000]}"
            output = llm(prompt, max_tokens=256, stop=["\n"])
            return output['choices'][0]['text'].strip()
        except Exception as e:
            return f"[LLAMA ERROR] {e}\n{text[:200]}..."
    else:
        # Use transformers pipeline
        try:
            from transformers import pipeline
            summarizer = pipeline("summarization", model=llm_model)
            # transformers models have a max token limit, so chunk if needed
            max_chunk = 1000
            chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
            summary = ""
            for chunk in chunks:
                out = summarizer(chunk, max_length=256, min_length=40, do_sample=False)
                summary += out[0]['summary_text'] + "\n"
            return summary.strip()
        except Exception as e:
            return f"[TRANSFORMERS ERROR] {e}\n{text[:200]}..."
