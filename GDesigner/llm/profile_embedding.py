from sentence_transformers import SentenceTransformer
import torch

# Cache model to avoid reloading
_cached_model = None

def get_sentence_embedding(sentence):
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Use CPU to avoid CUDA conflicts with main model
        _cached_model = _cached_model.to('cpu')
    
    # Clear CUDA cache before encoding to prevent conflicts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    embeddings = _cached_model.encode(sentence, device='cpu')
    return embeddings
