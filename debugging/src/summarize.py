def summarize_text(text, model_path, llm_model, n_ctx=4096, n_gpu_layers=48):
    """
    Summarize text using a GGUF LLM model with llama-cpp-python.
    Args:
        text (str): The input text to summarize.
        model_path (str): Path to the directory containing the model.
        llm_model (str): Filename of the GGUF model.
        n_ctx (int): Context window size.
        n_gpu_layers (int): Number of layers to run on GPU.
    Returns:
        str: The summary or error message.
    """
    import os
    try:
        from llama_cpp import Llama
        model_file = os.path.join(model_path, llm_model)
        llm = Llama(model_path=model_file, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        prompt = f"Resumí el siguiente texto en español de forma concisa y clara, usando viñetas si es posible.\n\nTexto:\n{text[:2000]}\n\nResumen:"
        output = llm(prompt, max_tokens=256)
        result = output['choices'][0]['text'].strip()
        if not result:
            print('[DEBUG] LLM returned empty summary. Full output:', output)
            return '[LLAMA ERROR] LLM returned empty summary.'
        print('[DEBUG] LLM summary:', result)
        return result
    except Exception as e:
        print(f'[DEBUG] Exception in LLM summarization: {e}')
        return f"[LLAMA ERROR] {e}\n{text[:200]}..."
