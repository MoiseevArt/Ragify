from typing import List, Optional, Dict, Any
from ragify.model import LLMModel
from tqdm import tqdm
import numpy as np
import torch


class ContextualChunking:
    def __init__(self,
                 corpus: List[str],
                 chunks: List[str],
                 indexes: List[int],
                 model: Optional[LLMModel] = None,
                 custom_config: Optional[Dict[str, Any]] = None,
                 tokenizer_params: Optional[Dict[str, Any]] = None,
                 system_prompt: Optional[str] = None):
        """
        ContextualChunking generates a brief context for each chunk based on the full document.

        Args:
            corpus (List[str]): List of full documents.
            chunks (List[str]): List of document chunks.
            indexes (List[int]): List of indices mapping chunks to documents.
            model (Optional[LLMModel]): Custom language model instance (defaults to Vikhrmodels/QVikhr-2.5-1.5B-Instruct-SMPO).
            custom_config (Optional[Dict[str, Any]]): Custom generation config (defaults provided).
            tokenizer_params (Optional[Dict[str, Any]]): Custom tokenization parameters (defaults provided).
            system_prompt (Optional[str]): Custom system prompt (defaults provided).
        """

        self.__corpus = corpus
        self.__chunks = chunks
        self.__indexes = indexes

        # Default system prompt
        self.__system_prompt = system_prompt or """You are an assistant for improving search in a RAG system. Generate a brief context (2-3 sentences)  
        that explains how this text fragment relates to the main document. The context should include:  
        - Key topics of the document  
        - The role of this fragment in the overall context  
        - Important terms and connections  
        The response should be in the language used in the document."""

        # Default model (if none provided)
        self.__model = model or LLMModel(
            model_name="Vikhrmodels/QVikhr-2.5-1.5B-Instruct-SMPO",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )

        # Default generation config (if none provided)
        self.__custom_config = custom_config or {
            "max_new_tokens": 200,
            "temperature": 0.4,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }

        # Default tokenization config (if none provided)
        self.__tokenizer_params = tokenizer_params or {
            'truncation': True,
            'add_generation_prompt': True,
            'return_tensors': "pt"
        }

    def get_system_prompt(self) -> str:
        """Returns the current system prompt."""
        return self.__system_prompt

    def set_system_prompt(self, system_prompt: str) -> None:
        """Sets a new system prompt."""
        self.__system_prompt = system_prompt

    def run_contextual_chunking(self) -> np.ndarray:
        """
        Performs contextual chunking by generating contexts for each document chunk.

        Returns:
            np.ndarray: Array of chunks augmented with generated context.
        """
        result_chunks = []

        for doc_idx, chunk in tqdm(zip(self.__indexes, self.__chunks), total=len(self.__chunks)):
            full_document = self.__corpus[doc_idx]

            user_prompt = f"""<document>
            {full_document}
            </document>

            Chunk of document:
            <chunk>
            {chunk}
            </chunk>

            Context for this text fragment:"""

            context = self.__model.generate(
                system_prompt=self.__system_prompt,
                prompt=user_prompt,
                custom_config=self.__custom_config,
                tokenizer_params=self.__tokenizer_params
            )

            result_chunks.append(f"{context}\n{chunk}")

        return np.array(result_chunks)
