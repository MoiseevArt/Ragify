from llama_index.core.node_parser import SentenceSplitter
from typing import Optional
import pandas as pd
import numpy as np
import json


# Splits a single document into chunks using a specified splitter
def split_into_chunks(doc, splitter):
    return splitter.split_text(doc)


# Splits a list of documents into chunks and keeps track of original indices
def get_split(documents, splitter):
    output_with_indices = []
    for index, doc in enumerate(documents):
        chunks = split_into_chunks(doc, splitter)
        for chunk in chunks:
            output_with_indices.append((index, chunk))
    return output_with_indices


def auto_preprocessing(
        json_path: Optional[str] = None,
        data_from_data_frame: Optional[pd.DataFrame] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 250
):
    """
    Automatically processes the input data by loading from a JSON file or DataFrame,
    and **splits the text into chunks** for further processing.
    **Warning!** Data must have a specific format.

    Args:
        json_path (str, optional): Path to the JSON file containing the dataset.
        data_from_data_frame (DataFrame, optional): A pandas DataFrame containing the data.
        chunk_size (int, optional): Maximum size of each chunk in characters (default 1000).
        chunk_overlap (int, optional): Overlap between consecutive chunks in characters (default 250).

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
            indexes (np.ndarray): Array of indexes corresponding to each chunk.
            texts (np.ndarray): Array of chunked texts.
            metadata (Optional[np.ndarray]): Array of metadata (if available), else `None`.
    """
    if json_path is not None and isinstance(json_path, pd.DataFrame) and data_from_data_frame is None:
        data_from_data_frame = json_path
        json_path = None

    if json_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_from_json = json.load(f)

        if 'data' in data_from_json:
            data_from_user = pd.DataFrame(data_from_json['data'])
        else:
            raise ValueError("The provided JSON does not contain the key 'data'.")

    elif data_from_data_frame is not None:
        if 'data' in data_from_data_frame:
            raise ValueError("If your DataFrame contains a column named 'data', you should pass it as pd.DataFrame(data['data'])")
        else:
            data_from_user = data_from_data_frame

    else:
        raise ValueError("Either 'json_path' or 'data_from_data_frame' must be provided.")

    for col in ['title', 'context']:
        if col not in data_from_user.columns:
            raise ValueError(f"Missing required column '{col}' in the data.")

    # Combine title and context into a single text column
    data_from_user['text'] = data_from_user['title'] + '\n' + data_from_user['context']
    corpus = data_from_user['text'].values

    metadata = data_from_user['metadata'].values if 'metadata' in data_from_user.columns else None

    # Initialize the sentence splitter with the provided chunk size and overlap
    splitter_classic = SentenceSplitter(
        chunk_size=chunk_size,  # Maximum size of each chunk in characters
        chunk_overlap=chunk_overlap  # Overlap between chunks to preserve context
    )

    # Split the corpus into chunks using the provided splitter function
    corpus_split = get_split(corpus, splitter_classic)
    indexes = np.array([f for f, _ in corpus_split])
    texts = np.array([s for _, s in corpus_split])

    return indexes, texts, metadata
