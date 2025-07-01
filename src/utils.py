
import spacy
from typing import List

_nlp = None
def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('es_core_news_sm')
    return _nlp

def sentence_chunk_text(text: str, max_chunk: int = 2000) -> List[str]:
    """
    Split text into chunks at sentence boundaries using spaCy, each chunk <= max_chunk chars.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text)
    chunks = []
    current = ""
    for sent in doc.sents:
        s = sent.text.strip()
        if len(current) + len(s) + 1 <= max_chunk:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks

def filter_unrelated_sections(summary: str) -> str:
    """
    Remove unrelated or repeated sections from the summary output.
    For now, removes lines mentioning 'persona', 'compañía', or 'despedida'.
    """
    import re
    lines = summary.splitlines()
    filtered = []
    for line in lines:
        if re.search(r'(persona|compañía|despedida)', line, re.IGNORECASE):
            continue
        filtered.append(line)
    return '\n'.join(filtered)
