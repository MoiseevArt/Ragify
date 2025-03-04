from typing import List, Optional, Dict, Tuple
from .BiEncoderRetriever import BiEncoder
from FlagEmbedding import FlagReranker
from collections import defaultdict
from .BM25Retriever import BM25


class RRFHybridPlusReranker:
    def __init__(self,
                 corpus: List[str],
                 metadata: Optional[List[Dict]] = None,
                 indexes: Optional[List[int]] = None,
                 lang: str = 'en',
                 batch_size: int = 16,
                 method: str = 'bm25+',
                 delta: float = 1.5
                 ):
        """
        Initializes the hybrid retrieval system that combines BM25, BiEncoder, and Reciprocal Rank Fusion (RRF)  for bge-reranker.

        Args:
            corpus (List[str]): List of texts (can be either chunks or full texts).
            metadata (Optional[List[Dict]]): List of metadata for the documents.
            indexes (Optional[List[int]]): List of indices for chunks. Required only if chunking is used AND metadata is provided.
            lang (str): Language setting (Only English stopwords are supported by default; other languages like German, Dutch, French, Spanish, Portuguese, Italian, Russian, Swedish, Norwegian, and Chinese are supported).
            batch_size (int): Batch size for BiEncoder model.
            method (str): The BM25 method ('robertson', 'lucene', 'atire', 'bm25l', 'bm25+').
            delta (float): Smoothing parameter for bm25+ method.

        List of BiEncoder Models:
            - For English and other languages: Uses `BAAI/bge-m3` from the BGEM3 family.
            - For Russian: Uses `deepvk/USER-bge-m3`, a fine-tuned version of `bge-m3` for better Russian language support.
        """

        self.__reranker = FlagReranker(model_name_or_path='BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.__bm = BM25(corpus=corpus, metadata=metadata, indexes=indexes, lang=lang, method=method, delta=delta)
        self.__biencoder = BiEncoder(corpus=corpus, metadata=metadata, indexes=indexes, batch_size=batch_size, lang=lang)

    @staticmethod
    def __rrf_fusion(texts_1: List[str],
                     metadata_1: Optional[List[Dict]],
                     texts_2: List[str],
                     metadata_2: Optional[List[Dict]],
                     k: int = 8,
                     rrf_param: int = 60) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Perform Reciprocal Rank Fusion (RRF) to combine two sets of ranked texts based on their scores.
        """
        scores = defaultdict(float)
        metadata_map = {}

        # Loop through the first set of texts and assign RRF scores based on their rank
        for rank, text in enumerate(texts_1, start=1):
            scores[text] += 1 / (rank + rrf_param)
            if metadata_1 is not None:
                metadata_map[text] = metadata_1[rank - 1]  # Map metadata to text if available

        # Loop through the second set of texts and apply the same process
        for rank, text in enumerate(texts_2, start=1):
            scores[text] += 1 / (rank + rrf_param)
            if metadata_2 is not None and text not in metadata_map:
                metadata_map[text] = metadata_2[rank - 1]  # Map metadata from the second set if not already added

        sorted_texts = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        top_texts = sorted_texts[:k]

        # Get the corresponding metadata for the top texts, if metadata exists
        top_metadata = [metadata_map[text] for text in top_texts] if metadata_map else None

        return top_texts, top_metadata

    def retrieve(self,
                 query: str,
                 top_k_from_bm25: int = 20,
                 top_k_from_biencoder: int = 20,
                 final_k: int = 10,
                 top_k_from_rrf: int = None,
                 rrf_param: int = 60) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Retrieves top-k relevant documents by combining BM25, BiEncoder and RRF reranking and then re-ranks them with a neural reranker.

        Args:
            query (str): The query string to search for.
            top_k_from_bm25 (int): Number of top-k documents to retrieve using BM25 method.
            top_k_from_biencoder (int): Number of top-k documents to retrieve using BiEncoder method.
            final_k (int): Number of final top-k documents to return after reranking.
            top_k_from_rrf (Optional[int]): Number of top-k documents after RRF fusion. Default is None, which means 80% of combined BM25 and BiEncoder results.
            rrf_param (int): RRF parameter that influences the fusion formula. It is used in the calculation of the reciprocal rank score, and can be seen as a form of scaling factor.

        Returns:
            Tuple[List[str], Optional[List[Dict]]]: Retrieved texts and corresponding metadata (if available).
        """
        if top_k_from_rrf is not None and final_k > top_k_from_rrf:
            raise ValueError(f"Requested 'final_k' ({final_k}) cannot be greater than 'top_k_from_rrf' ({top_k_from_rrf}).")

        if top_k_from_rrf is None:
            top_k_from_rrf = int(0.8 * (top_k_from_bm25 + top_k_from_biencoder))

        if final_k > top_k_from_rrf:
            raise ValueError(
                f"Requested 'final_k' ({final_k}) cannot be greater than 80% of top_k_from_bm25 + top_k_from_biencoder ({top_k_from_rrf}).")

        be_texts, be_metadata = self.__biencoder.retrieve(query=query, k=top_k_from_bm25)
        bm_texts, bm_metadata = self.__bm.retrieve(query=query, k=top_k_from_biencoder)

        rrf_texts, rrf_metadata = self.__rrf_fusion(texts_1=be_texts, metadata_1=be_metadata, texts_2=bm_texts, metadata_2=bm_metadata, k=top_k_from_rrf, rrf_param=rrf_param)

        # Prepare query-document pairs for the reranker
        pairs = [(query, doc) for doc in rrf_texts]
        scores = self.__reranker.compute_score(pairs)

        # Sort documents by reranker scores in descending order
        sorted_results = sorted(zip(scores, rrf_texts, rrf_metadata or [None] * len(rrf_texts)), reverse=True)

        sorted_texts = [doc for _, doc, _ in sorted_results]
        sorted_metadata = [meta for _, _, meta in sorted_results] if rrf_metadata else None

        return sorted_texts[:final_k], sorted_metadata[:final_k] if sorted_metadata else None
