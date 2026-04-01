import os
import sys


_ALIAS_TO_CANONICAL = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bgebase": "BAAI/bge-base-en-v1.5",
    "baai/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bgelarge": "BAAI/bge-large-en-v1.5",
    "baai/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
    "e5base": "intfloat/e5-base-v2",
    "e5bse": "intfloat/e5-base-v2",
    "intfloat/e5-base-v2": "intfloat/e5-base-v2",
    "e5-large": "intfloat/e5-large-v2",
    "e5large": "intfloat/e5-large-v2",
    "intfloat/e5-large-v2": "intfloat/e5-large-v2",
    "gte-base": "thenlper/gte-base",
    "gtebase": "thenlper/gte-base",
    "thenlper/gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",
    "gtelarge": "thenlper/gte-large",
    "thenlper/gte-large": "thenlper/gte-large",
}

_CANONICAL_TO_TAG = {
    "sentence-transformers/all-mpnet-base-v2": "mpnet",
    "BAAI/bge-base-en-v1.5": "bge_base",
    "BAAI/bge-large-en-v1.5": "bge_large",
    "intfloat/e5-base-v2": "e5_base",
    "intfloat/e5-large-v2": "e5_large",
    "thenlper/gte-base": "gte_base",
    "thenlper/gte-large": "gte_large",
}

_CANONICAL_TO_DIM = {
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
}


def resolve_text_model_name(text_model):
    model_key = str(text_model).strip()
    if not model_key:
        model_key = "mpnet"
    return _ALIAS_TO_CANONICAL.get(model_key, _ALIAS_TO_CANONICAL.get(model_key.lower(), model_key))


def text_model_tag(text_model):
    canonical_name = resolve_text_model_name(text_model)
    if canonical_name in _CANONICAL_TO_TAG:
        return _CANONICAL_TO_TAG[canonical_name]
    return canonical_name.replace("/", "_").replace("-", "_").lower()


def text_model_embedding_dim(text_model):
    canonical_name = resolve_text_model_name(text_model)
    return _CANONICAL_TO_DIM.get(canonical_name)


def resolve_embed_size(requested_embed_size, text_model):
    model_dim = text_model_embedding_dim(text_model)
    if model_dim is None:
        return requested_embed_size

    # Auto-upgrade only when parser default (768) would break for larger models.
    if requested_embed_size == 768 and model_dim != 768:
        return model_dim
    return requested_embed_size


def embedding_dir(base_dir, text_model):
    return os.path.join(base_dir, text_model_tag(text_model))


def embedding_file(base_dir, text_model, file_name):
    return os.path.join(embedding_dir(base_dir, text_model), file_name)


def resolve_embedding_file(base_dir, text_model, file_name):
    model_scoped_path = embedding_file(base_dir, text_model, file_name)
    if os.path.exists(model_scoped_path):
        return model_scoped_path
    # Backward compatibility with older layout under data/*.h5
    return os.path.join(base_dir, file_name)


def build_log_path(log_dir, script_stem, method, dataset, text_model):
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(
        log_dir,
        f"{script_stem}_{method}_{text_model_tag(text_model)}_{dataset}.log",
    )


class TeeStream:
    def __init__(self, stream_a, stream_b):
        self.stream_a = stream_a
        self.stream_b = stream_b

    def write(self, data):
        self.stream_a.write(data)
        self.stream_b.write(data)

    def flush(self):
        self.stream_a.flush()
        self.stream_b.flush()


def setup_run_logging(log_path):
    log_fp = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_fp)
    sys.stderr = TeeStream(sys.stderr, log_fp)
    return log_fp