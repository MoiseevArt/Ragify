from typing import List, Optional, Dict, Tuple
import warnings
import Stemmer
import bm25s


class BM25:
    def __init__(self,
                 corpus: List[str],
                 metadata: Optional[List[Dict]] = None,
                 indexes: Optional[List[int]] = None,
                 lang: str = 'en',
                 method: str = 'bm25+',
                 delta: float = 1.5):
        """
        Initializes a BM25 retriever capable of handling both chunks of text and full texts.

        Args:
            corpus (List[str]): List of texts (can be either chunks or full texts).
            metadata (Optional[List[Dict]]): List of metadata for the documents.
            indexes (Optional[List[int]]): List of indices for chunks. Required only if chunking is used AND metadata is provided.
            lang (str): Language for stemming and stopwords (Only English stopwords are supported by default; other languages like German, Dutch, French, Spanish, Portuguese, Italian, Russian, Swedish, Norwegian, and Chinese are supported).
            method (str): The BM25 method ('robertson', 'lucene', 'atire', 'bm25l', 'bm25+').
            delta (float): Smoothing parameter for bm25+ method.
        """
        self.__lang = lang
        self.__texts = corpus
        self.__indexes = indexes
        self.__metadata = metadata
        self.__stemmer = Stemmer.Stemmer(lang)
        self.__retriever = bm25s.BM25(method=method, delta=delta)
        self.__text_tokens = None
        self.__index_corpus()

    def __index_corpus(self):
        """Indexes the provided texts."""
        self.__text_tokens = bm25s.tokenize(self.__texts, stemmer=self.__stemmer, stopwords=self.__lang)
        self.__retriever.index(self.__text_tokens)

    def retrieve(self,
                 query: str,
                 k: int = 10) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Retrieves top-k for a given query.

        Args:
            query (str): The query string to search for.
            k (int): The number of results to return.

        Returns:
            Tuple[List[str], Optional[List[Dict]]]: Retrieved texts and corresponding metadata (if available).
        """

        if k > len(self.__texts):
            raise ValueError(f"Requested 'k' ({k}) exceeds the number of available texts ({len(self.__texts)}).")

        query_tokens = bm25s.tokenize(query, stemmer=self.__stemmer, stopwords=self.__lang)
        result_indexes, _ = self.__retriever.retrieve(query_tokens, k=k)

        result_indexes = result_indexes[0]
        best_texts = [self.__texts[idx] for idx in result_indexes]

        if self.__metadata is not None:
            if self.__indexes is not None:
                best_metadata = [self.__metadata[self.__indexes[idx]] for idx in result_indexes]
            else:
                if len(self.__texts) != len(self.__metadata):
                    warnings.warn(
                        "It looks like you are using chunked texts. For correct metadata handling, 'indexes' should be provided.",
                        UserWarning)
                best_metadata = [self.__metadata[idx] for idx in result_indexes]
        else:
            best_metadata = None

        return best_texts, best_metadata
