"""Explicit embedding selection; provider failure never switches models."""


def create_embedder(settings, *, download=False, request_limiter=None):
    if settings.embedding_provider=='gemini':
        from .gemini import GeminiEmbedder
        return GeminiEmbedder(settings,request_limiter=request_limiter)
    from .embeddings import LocalEmbedder
    return LocalEmbedder(settings.runtime_dir/'models',allow_download=download)


def index_directory(settings):
    return settings.runtime_dir/('retrieval_gemini' if settings.embedding_provider=='gemini' else 'retrieval')
