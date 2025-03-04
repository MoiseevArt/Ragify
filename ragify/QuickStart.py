from typing import List, Optional, Dict, Tuple
from ragify.model import LLMModel
from ragify.retrievers import *
import torch
import re


class FastRAG:
    """
    FastRAG is a utility class designed for rapidly bootstrapping a Retrieval-Augmented Generation (RAG) system.
    It supports three operational modes—light, medium, and hard—each representing a different level of resource
    consumption and performance. This class seamlessly integrates retrieval and generation components, allowing you
    to quickly deploy a RAG pipeline tailored to your specific resource constraints and application needs.
    """

    def __init__(self,
                 corpus: List[str],
                 metadata: Optional[List[Dict]] = None,
                 indexes: Optional[List[int]] = None,
                 lang: str = 'en') -> None:
        """
        Initialize a FastRAG instance with the necessary corpus and configuration settings.

        This initializer sets up the foundational parameters for the RAG system, including the corpus of texts,
        optional metadata, and language configuration. After initialization, you can configure the system to operate
        in one of three modes (light, medium, or hard) based on your desired balance between performance and resource usage.

        Args:
            corpus (List[str]): A list of texts that form the knowledge base for retrieval.
            metadata (Optional[List[Dict]]): An optional list of dictionaries containing metadata for each text.
            indexes (Optional[List[int]]): An optional list of indexes corresponding to the texts in the corpus.
            lang (str): The language of the texts.
        """
        self.__lang: str = 'ru' if lang.lower() == 'russian' else lang.lower()
        self.__texts: List[str] = corpus
        self.__indexes: Optional[List[int]] = indexes
        self.__metadata: Optional[List[Dict]] = metadata
        self.__method = None
        self.__model = None
        self.__mode = None
        self.__retriever_param = {}
        self.__model_config: Dict = {
            "max_new_tokens": 1000,
            "num_return_sequences": 1
        }
        self.__tokenizer_params: Dict = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "max_length": 8000
        }
        self.__system_prompt: str = (
            "You are a RAG assistant that answers the user's question based on the texts provided to you. "
            "The user does not see the texts that I send you, so do not mention them. There will be a total of 5 texts, "
            "and at the end of your response, indicate the indexes of the texts you used the most for your answer, just indexes separated by spaces!"
            "Formulate your answer in the same language as the texts! If there is no answer in the texts, refuse to answer."
        )
        self.__light_system_prompt: str = (
            "You are a RAG assistant that answers the user's question based on the texts provided to you. "
            "Formulate your answer in the same language as the texts! If there is no answer in the texts, refuse to answer."
        )

    def set_mode(self, mode: str,
                 attn_implementation: Optional[str] = None) -> None:
        """
        Set the operational mode, initializing the corresponding retriever and language model.

        This method configures the RAG system to operate in one of three predefined modes:
        - **light**: Low resource consumption, using BM25 and Reranker for retrieval and a lightweight LLM.
        - **medium**: Balanced performance, utilizing BiEncoder and Reranker for retrieval and a mid-sized LLM.
        - **hard**: High resource consumption, employing RRFHybrid and Reranker for advanced retrieval and a large-scale LLM.

        Args:
            mode (str): The mode of operation. Must be one of 'light', 'medium', or 'hard'.
            attn_implementation (Optional[str]): Optional attention implementation setting for the language model.

        Raises:
            ValueError: If an unknown mode is provided.

        **light**:
            - Retriever: `BM25PlusReranker`
            - Model: `"Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct"`

        **medium**:
            - Retriever: `BiEncoderPlusReranker`
            - Model: `"IlyaGusev/saiga_gemma2_10b"` (for Russian) and `"Qwen/Qwen2-7B-Instruct"` (for other languages)

        **hard**:
            - Retriever: `RRFHybridPlusReranker`
            - Model: `"Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"`
        """
        if mode == "light":
            self.__method = BM25PlusReranker(
                corpus=self.__texts,
                indexes=self.__indexes,
                metadata=self.__metadata,
                lang=self.__lang
            )
            self.__model = LLMModel(
                model_name="Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attn_implementation
            )
            self.__retriever_param = {"k2": 5}
            self.__mode = mode

        elif mode == "medium":
            self.__method = BiEncoderPlusReranker(
                corpus=self.__texts,
                indexes=self.__indexes,
                metadata=self.__metadata,
                lang=self.__lang
            )
            if self.__lang == 'ru':
                self.__model = LLMModel(
                    model_name="IlyaGusev/saiga_gemma2_10b",
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            else:
                self.__model = LLMModel(
                    model_name="Qwen/Qwen2-7B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation=attn_implementation
                )
            self.__retriever_param = {"k2": 5}
            self.__mode = mode

        elif mode == "hard":
            self.__method = RRFHybridPlusReranker(
                corpus=self.__texts,
                indexes=self.__indexes,
                metadata=self.__metadata,
                lang=self.__lang,
                batch_size=8
            )
            self.__model = LLMModel(
                model_name="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attn_implementation
            )
            self.__retriever_param = {"final_k": 5}
            self.__mode = mode
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run(self, query: str) -> Tuple[str, Optional[List[Dict]]]:
        """
        Process the user query by retrieving relevant texts and generating an answer.

        Args:
            query (str): The user's question.

        Returns:
            Tuple[str, Optional[List[Dict]]]: A tuple containing:
                - The generated answer (str).
                - The corresponding metadata (if available).

        Raises:
            RuntimeError: If the mode has not been set prior to calling run().
        """
        if self.__method is None or self.__model is None:
            raise RuntimeError("Mode is not set! Call `set_mode(mode)` before `run()`.")

        texts, metadata = self.__method.retrieve(query=query, **self.__retriever_param)
        prompt = self.__build_prompt(query, texts)
        system_prompt = self.__light_system_prompt if self.__mode == "light" else self.__system_prompt

        generated_output: str = self.__model.generate(prompt=prompt,
                                                      system_prompt=system_prompt,
                                                      custom_config=self.__model_config,
                                                      tokenizer_params=self.__tokenizer_params)

        if self.__mode == "light":
            return generated_output, None

        final_output, new_metadata = self.__process_output(generated_output, metadata)
        return final_output, new_metadata

    @staticmethod
    def __build_prompt(query: str, texts: List[str]) -> str:
        """
        Construct the prompt string for the language model based on the retrieved texts and the user's query.
        """
        prompt_parts = [f"Text {i}:\n{text}\n" for i, text in enumerate(texts)]
        prompt_parts.append(f"question:{query}\nYour answer:")
        return "\n".join(prompt_parts)

    @staticmethod
    def __process_output(output: str,
                         metadata: Optional[List[Dict]]) -> Tuple[str, Optional[List[Dict]]]:
        """
        Process the generated output to extract the final answer and filter metadata based on extracted indexes.

        The output is split into sentences. If the last sentence contains indices (digits 0-4), it is removed from the final answer.
        The extracted indices are then used to select the corresponding metadata entries.
        """
        sentences = re.split(r'(?<=[.!?])\s+', output.strip())
        new_metadata: Optional[List[Dict]] = None

        if len(sentences) > 1:
            last_sentence = sentences[-1]
            indices = [int(num) for num in re.findall(r'[0-4]', last_sentence)]
            if indices:
                output = ' '.join(sentences[:-1])
                if metadata:
                    new_metadata = [metadata[i] for i in indices if i < len(metadata)]
        return output, new_metadata
