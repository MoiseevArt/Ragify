from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Tuple
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import warnings
import faiss


class BiEncoder:
    def __init__(self,
                 corpus: List[str],
                 metadata: Optional[List[Dict]] = None,
                 indexes: Optional[List[int]] = None,
                 batch_size: int = 16,
                 lang: str = 'en'):
        """
        Initializes a BiEncoder retriever using FAISS for efficient similarity search capable of handling both chunks of text and full texts.

        Args:
            corpus (List[str]): List of texts (can be either chunks or full texts).
            metadata (Optional[List[Dict]]): List of metadata for the documents.
            indexes (Optional[List[int]]): List of indices for chunks. Required only if chunking is used AND metadata is provided.
            batch_size (int): Batch size for encoding model.
            lang (str): Language setting.

        Encoding Models:
            - For English and other languages: Uses `BAAI/bge-m3` from the BGEM3 family.
            - For Russian: Uses `deepvk/USER-bge-m3`, a fine-tuned version of `bge-m3` for better Russian language support.
        """

        self.__lang = 'ru' if lang.lower() == 'russian' else lang.lower()
        self.__texts = list(corpus)
        self.__indexes = indexes
        self.__metadata = metadata
        self.__batch_size = batch_size
        self.__faiss = self.__get_faiss()

    def __get_faiss(self):
        """
        Creates a FAISS index from encoded text embeddings.
        """
        if self.__lang == 'ru':
            self.__model = SentenceTransformer("deepvk/USER-bge-m3")  # Load the Russian bi-encoder model

            embeddings = self.__model.encode(self.__texts, batch_size=self.__batch_size)
        else:
            self.__model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # Load the general multi-language bi-encoder model

            # Encode texts and extract dense embeddings
            embeddings_dict = self.__model.encode(self.__texts, batch_size=self.__batch_size)
            embeddings = embeddings_dict["dense_vecs"]
            embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]  # Determine embedding dimension

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index

    def retrieve(self,
                 query: str,
                 k: int = 10) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Retrieves top-k most relevant texts for a given query.

        Args:
            query (str): The query string to search for.
            k (int): The number of results to return.

        Returns:
            Tuple[List[str], Optional[List[Dict]]]: Retrieved texts and corresponding metadata (if available).
        """

        query_embedding = self.__model.encode([query], batch_size=1)

        if self.__lang != 'ru':
            # Convert query embedding to NumPy array if not using the Russian model
            query_embedding = np.array(query_embedding["dense_vecs"], dtype=np.float32)

        # Search FAISS index for top-k most similar embeddings
        _, result_indexes = self.__faiss.search(query_embedding, k)
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
