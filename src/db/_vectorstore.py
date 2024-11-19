from llama_index.vector_stores.postgres import PGVectorStore

from src.config.configuration import db_config


def get_vector_store(
    table_name: str = "dev_vectors",  # TODO: adjust
    embed_dim: int = 384,  # TODO: adjust
) -> PGVectorStore:
    """
    Connect to the postgres database and set up PGVectorStore.

    Args:
        table_name (str): The table name. Defaults to "dev_vectors".
        embed_dim (int): The embedding dimension. Defaults to 384.

    Returns:
        PGVectorStore: The vector store.
    """
    # Connect to the new database and set up PGVectorStore
    vector_store = PGVectorStore.from_params(
        database=db_config.db_name,
        host=db_config.host,
        password=db_config.password,
        port=db_config.port,
        user=db_config.user,
        table_name=table_name,
        embed_dim=embed_dim,  # Adjust as needed for your embeddings
    )

    return vector_store
