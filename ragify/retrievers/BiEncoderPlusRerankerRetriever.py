from typing import List, Optional, Dict, Tuple
from FlagEmbedding import FlagReranker
from .BiEncoderRetriever import BiEncoder


class BiEncoderPlusReranker:
    def __init__(self,
                 corpus: List[str],
                 metadata: Optional[List[Dict]] = None,
                 indexes: Optional[List[int]] = None,
                 batch_size: int = 16,
                 lang: str = 'en'):
        """
        Initializes a BiEncoder retriever for bge-reranker capable of handling both chunks of text and full texts.
        Using FAISS for efficient similarity search.

        Args:
            corpus (List[str]): List of texts (can be either chunks or full texts).
            metadata (Optional[List[Dict]]): List of metadata for the documents.
            indexes (Optional[List[int]]): List of indices for chunks. Required only if chunking is used AND metadata is provided.
            batch_size (int): Batch size for encoding model.
            lang (str): Language setting.

        List of Encoding Models:
            - For English and other languages: Uses `BAAI/bge-m3` from the BGEM3 family.
            - For Russian: Uses `deepvk/USER-bge-m3`, a fine-tuned version of `bge-m3` for better Russian language support.
        """

        self.__reranker = FlagReranker(model_name_or_path='BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.__biencoder = BiEncoder(corpus=corpus, metadata=metadata, indexes=indexes, batch_size=batch_size, lang=lang)

    def retrieve(self,
                 query: str,
                 k1: int = 20,
                 k2: int = 10) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Retrieves documents using BiEncoder and then re-ranks them with a neural reranker.

        Args:
            query (str): The query string to search for.
            k1 (int): The number of top documents retrieved by BiEncoder before reranking.
            k2 (int): The final number of documents returned after reranking.

        Returns:
            Tuple[List[str], Optional[List[Dict]]]: Retrieved texts and corresponding metadata (if available).

        Raises:
            ValueError: If `k2` is greater than `k1`, since reranking cannot return more documents than BM25 retrieves.
        """

        if k2 > k1:
            raise ValueError(f"Requested 'k2' ({k2}) cannot be greater than 'k1' ({k1}).")

        # Retrieve top-k1 documents from BiEncoder
        be_texts, be_metadata = self.__biencoder.retrieve(query=query, k=k1)

        # Prepare query-document pairs for the reranker
        pairs = [(query, doc) for doc in be_texts]
        scores = self.__reranker.compute_score(pairs)

        # Sort documents by reranker scores in descending order
        sorted_results = sorted(zip(scores, be_texts, be_metadata or [None] * len(be_texts)), reverse=True)

        sorted_texts = [doc for _, doc, _ in sorted_results]
        sorted_metadata = [meta for _, _, meta in sorted_results] if be_metadata else None

        return sorted_texts[:k2], sorted_metadata[:k2] if sorted_metadata else None
