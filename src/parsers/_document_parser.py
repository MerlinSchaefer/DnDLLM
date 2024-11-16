from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.readers.file import MarkdownReader, PyMuPDFReader

from src.llm.embeddings import get_embedding_model


# TODO: look into MarkdownNodeParser
class DocumentParser:
    def __init__(self, file_path: str, doc_type: str, chunk_size: int = 1024):
        """
        Initialize the DocumentParser.

        Args:
            file_path (str): The path to the document file.
            doc_type (str): The type of the document ('pdf' or 'markdown').
            chunk_size (int): The size of the text chunks. Defaults to 1024.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.text_parser = SentenceSplitter(chunk_size=self.chunk_size)
        self.documents: list = []
        self.text_chunks: list = []
        self.doc_idxs: list = []
        self.nodes: list = []

        if doc_type == "pdf":
            self.loader = PyMuPDFReader()
        elif doc_type == "markdown":
            self.loader = MarkdownReader()
        else:
            raise ValueError("Unsupported document type. Use 'pdf' or 'markdown'.")

    def _read_documents(self) -> None:
        """
        Read documents from the file.
        """
        self.documents = self.loader.load(file_path=self.file_path)

    def _split_into_chunks(self) -> list[str]:
        """
        Split the documents into text chunks.

        Returns:
            list[str]: The list of text chunks.
        """
        for doc_idx, doc in enumerate(self.documents):
            cur_text_chunks = self.text_parser.split_text(doc.text)
            self.text_chunks.extend(cur_text_chunks)
            self.doc_idxs.extend([doc_idx] * len(cur_text_chunks))
        return self.text_chunks

    def _create_nodes(self) -> list[TextNode]:
        """
        Create nodes from the text chunks.

        Returns:
            list[TextNode]: The list of text nodes.
        """
        for idx, text_chunk in enumerate(self.text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = self.documents[self.doc_idxs[idx]]
            node.metadata = src_doc.metadata
            self.nodes.append(node)
        return self.nodes

    def _embed_nodes(self) -> None:
        """
        Embed the nodes using the embedding model.
        """
        embed_model = get_embedding_model()
        for node in self.nodes:
            node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
            node.embedding = node_embedding

    def load_chunk_and_embed(self) -> list[TextNode]:
        """
        Load the document, split it into chunks, create nodes, and embed the nodes.

        Returns:
            list[TextNode]: The list of embedded text nodes.
        """
        self._read_documents()
        self._split_into_chunks()
        self._create_nodes()
        self._embed_nodes()
        return self.nodes


if __name__ == "__main__":
    # Usage example
    file_path = "../../../data/llama2.pdf"
    doc_type = "pdf"
    document_parser = DocumentParser(file_path, doc_type)
    nodes = document_parser.load_chunk_and_embed()
