from enum import Enum
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

LOCAL_MODEL_CACHE = Path(__file__).resolve().parent.parent / ".model_cache"


class AvailableEmbeddingModels(Enum):
    BGE_SMALL_EN = "BAAI/bge-small-en"
    BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"
    # Add other models here as needed


def get_embedding_model(
    model_name: AvailableEmbeddingModels = AvailableEmbeddingModels.BGE_SMALL_EN,
) -> HuggingFaceEmbedding:
    """
    Get the embedding model.

    Args:
        model_name (AvailableEmbeddingModels): The model name to use.

    Returns:
        HuggingFaceEmbedding: The embedding model.
    """
    return HuggingFaceEmbedding(model_name=model_name.value, cache_folder=str(LOCAL_MODEL_CACHE))
