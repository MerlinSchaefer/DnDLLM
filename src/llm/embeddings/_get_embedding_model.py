# sentence transformers
from enum import Enum

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class AvailableEmbeddingModels(Enum):
    BGE_SMALL_EN = "BAAI/bge-small-en"
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
    return HuggingFaceEmbedding(model_name=model_name.value, cache_folder="../.model_cache")
