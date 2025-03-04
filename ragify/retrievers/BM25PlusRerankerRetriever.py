from typing import List, Optional, Dict, Tuple
from FlagEmbedding import FlagReranker
from .BM25Retriever import BM25


class BM25PlusReranker:
    def __init__(self,
                 corpus: List[str],
                 metadata: Optional[List[Dict]] = None,
                 indexes: Optional[List[int]] = None,
                 lang: str = 'en',
                 method: str = 'bm25+',
                 delta: float = 1.5):
        """
        Initializes a BM25 retriever for bge-reranker capable of handling both chunks of text and full texts.

        Args:
            corpus (List[str]): List of texts (can be either chunks or full texts).
            metadata (Optional[List[Dict]]): List of metadata for the documents.
            indexes (Optional[List[int]]): List of indices for chunks. Required only if chunking is used AND metadata is provided.
            lang (str): Language for stemming and stopwords (Only English stopwords are supported by default; other languages like German, Dutch, French, Spanish, Portuguese, Italian, Russian, Swedish, Norwegian, and Chinese are supported).
            method (str): The BM25 method ('robertson', 'lucene', 'atire', 'bm25l', 'bm25+').
            delta (float): Smoothing parameter for bm25+ method.
        """

        self.__reranker = FlagReranker(model_name_or_path='BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.__bm = BM25(corpus=corpus, indexes=indexes, metadata=metadata, lang=lang, method=method, delta=delta)

    def retrieve(self,
                 query: str,
                 k1: int = 20,
                 k2: int = 10) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Retrieves documents using BM25 and then re-ranks them with a neural reranker.

        Args:
            query (str): The query string to search for.
            k1 (int): The number of top documents retrieved by BM25 before reranking.
            k2 (int): The final number of documents returned after reranking.

        Returns:
            Tuple[List[str], Optional[List[Dict]]]: Retrieved texts and corresponding metadata (if available).

        Raises:
            ValueError: If `k2` is greater than `k1`, since reranking cannot return more documents than BM25 retrieves.
        """

        if k2 > k1:
            raise ValueError(f"Requested 'k2' ({k2}) cannot be greater than 'k1' ({k1}).")

        # Retrieve top-k1 documents from BM25
        bm_texts, bm_metadata = self.__bm.retrieve(query=query, k=k1)

        # Prepare query-document pairs for the reranker
        pairs = [(query, doc) for doc in bm_texts]
        scores = self.__reranker.compute_score(pairs)

        # Sort documents by reranker scores in descending order
        sorted_results = sorted(zip(scores, bm_texts, bm_metadata or [None] * len(bm_texts)), reverse=True)

        sorted_texts = [doc for _, doc, _ in sorted_results]
        sorted_metadata = [meta for _, _, meta in sorted_results] if bm_metadata else None

        return sorted_texts[:k2], sorted_metadata[:k2] if sorted_metadata else None
