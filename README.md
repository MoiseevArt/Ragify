![logo](https://imgur.com/JwdgMVh.png)

---
## Table of Contents
1. [Introduction](#1-introduction-lizard)
2. [Framework Structure](#2-framework-structure-jigsaw)
3. [Evaluation Results](#3-evaluation-results)
4. [How to use](#4-how-to-use)
5. [Installation](#5-installation-inbox_tray)
6. [License](#6-license)

## 1. Introduction :lizard:
Retrieval-Augmented Generation (RAG) - is a technique that combines information retrieval with generative models to improve the quality and relevance of generated text. Instead of relying solely on pre-trained knowledge, RAG dynamically fetches relevant documents from a database and uses them as additional context for generation.

**Ragify** - is a simple and accessible framework that provides a variety of retrievers and ready-to-use RAG implementations tailored to different requirements. You can either build a custom RAG pipeline by selecting specific components or use one of the pre-configured solutions for a quick start. The framework is designed to be modular, flexible, and easy to integrate into existing workflows.

---
## 2. Framework Structure :jigsaw:

Ragify provides various modules for the flexible construction, such as:

### Retrievers:
- **BM25:** A sparse retriever based on term frequency-inverse document frequency (TF-IDF). It is lightweight, efficient, and works well for keyword-based retrieval tasks. Ideal for cases where keyword matching is sufficient.

- **BM25 + Reranker:** Enhances BM25 by applying a reranker, improving result quality. The reranker refines the ranking based on semantic relevance. Recommended when keyword matching alone is not enough.

- **BiEncoder:** A dense retriever using embeddings from BiEncoder. It provides better semantic search capabilities compared to BM25. Useful for cases requiring deeper understanding of text meaning.

- **BiEncoder + Reranker:** A combination of BiEncoder retrieval and reranking. The reranker further improves the ranking of retrieved documents, ensuring higher relevance. Best for scenarios where retrieval precision is crucial.

- **RRF (BM25, BiEncoder) + Reranker:** Uses Reciprocal Rank Fusion (RRF) to combine results from BM25 and BiEncoder before applying reranking. This hybrid approach balances keyword-based and semantic retrieval, making it highly effective for diverse queries.

Metadata-aware Reranking: Rerankers can process not only text but also metadata, ensuring that relevant metadata fields are considered in ranking results.
> [!NOTE]
> Used reranker: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
> Used BiEncoder: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) or [deepvk/USER-bge-m3](https://huggingface.co/deepvk/USER-bge-m3) for the Russian language.

---
### Model:
A key component of the RAG pipeline is the Large Language Model (LLM), which processes the retrieved documents and generates a coherent, context-aware response.

Ragify includes the LLMModelInterface, which allows users to load any model from Hugging Face, providing flexibility in choosing the most suitable language model for their needs. Users can also configure the model according to their specific requirements, adjusting parameters and behaviors to optimize performance for their use case.

---
### Utils:
- Automatic chunking function for efficiently preparing document corpora.
- Contextual Chunking. Contextual Chunking - is an innovative method introduced by [Anthropic](https://www.anthropic.com/news/contextual-retrieval). It involves generating additional context for each data chunk, enhancing the quality of information retrieval. This method reduces retrieval errors by approximately 1.5 times compared to traditional approaches, improving the overall performance of retrieval-based models in RAG setups.
> [!NOTE]
> Used model for contextual chunking: [Vikhrmodels/QVikhr-2.5-1.5B-Instruct-SMPO](https://huggingface.co/Vikhrmodels/QVikhr-2.5-1.5B-Instruct-SMPO).

---
### Quick Start: :zap:

The Quick Start module allows users to easily launch a pre-configured RAG setup. It offers three execution modes: light, medium, and hard. Essentially, this module provides a ready-to-use RAG system, making it simple to get started.
modes:
- Light: Used **BM25 + Reranker** retriever with [Vikhr-Qwen-2.5-1.5B](https://huggingface.co/Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct) model. This mode is the simplest and least resource-intensive. It provides a basic RAG configuration, ideal for quick testing or smaller-scale tasks with minimal computational requirements.

- Medium: Used **BiEncoder + Reranker** retriever with [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct) or [saiga_gemma2_10b](https://huggingface.co/IlyaGusev/saiga_gemma2_9b) for Russian. The medium mode offers a more advanced configuration. It strikes a balance between performance and resource usage, suitable for more demanding tasks without overloading the system.

- Hard: Used **RRF (BM25, BiEncoder) + Reranker** retriever with [Vikhr-Nemo-12B](https://huggingface.co/Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24) model. This is the most complex configuration, designed for high-performance use cases. It provides the most powerful setup but requires more computational resources, making it ideal for large-scale or resource-intensive tasks.
> [!NOTE]
> This solution is also capable of processing metadata.
---
## 3. Evaluation Results
Results on the [MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco) benchmark:
| Retriever                        | Precision@3 | Recall@3 | MRR      | nDCG@3   | MAP      |
|----------------------------------|-------------|----------|----------|----------|----------|
| BM25                             | 0.6724      | 0.6717   | 0.8361   | 0.6998   | 0.827    |
| BM25 + Reranker                  | 0.7908      | 0.7893   | 0.9107   | 0.8128   | 0.9048   |
| BiEncoder                        | 0.7990      | 0.7976   | 0.9124   | 0.8178   | 0.9060   |
| BiEncoder + Reranker             | 0.8439      | 0.8418   | 0.9262   | 0.8558   | 0.9199   |
| RRF (BM25, BiEncoder) + Reranker | **0.8469**  |**0.8472**|**0.9273**|**0.8583**|**0.9208**|

| Retriever                        | Precision@5 | Recall@5 | MRR      | nDCG@5   | MAP      |
|----------------------------------|-------------|----------|----------|----------|----------|
| BM25                             | 0.8397      | 0.8412   | 0.9774   | 0.87     | 0.9518   |
| BM25 + Reranker                  | 0.9099      | 0.9099   | 0.9847   | 0.9307   | 0.9834   |
| BiEncoder                        | 0.9298      | 0.9298   | 0.9784   | 0.9420   | 0.9745   |
| BiEncoder + Reranker             | 0.9481      | 0.9481   | 0.9707   | 0.9533   | 0.9701   |
| RRF (BM25, BiEncoder) + Reranker | **0.9527**  |**0.9542**|**0.9835**|**0.9594**|**0.9808**|

Results on the [SQUAD](https://huggingface.co/datasets/rajpurkar/squad) benchmark:
| model               | BERT Score |
|---------------------|------------|
| Vikhr-Qwen-2.5-1.5B | 0.7740     |
| Qwen2-7B            | 0.8069     |
| saiga_gemma2_10b    | 0.8281     |
| Vikhr-Nemo-12B      |**0.8813**  |

---
## 4. How to use
### Quick Start
If you don't want to configure the RAG setup yourself, use the ready-made implementation.

> [!NOTE]
> For better scalability of your RAG system, it is recommended to split the input text into chunks. The minimum parameter is the text corpus. If you are using chunks with metadata, you should also provide a list of indices for the chunks.

```python
from ragify import FastRAG

rag = FastRAG(corpus=texts, metadata=metadata, indexes=indexes, lang='en')
rag.set_mode('medium')

result, result_metadata = rag.run(query="what is happening in montreal in august?")
```
---
### Retrievers
To build your own RAG, you can combine different retrievers & models. Here's an example of using one of the retrievers.
```python
from ragify.retrievers import BiEncoder

rag = BiEncoder(corpus=texts,
                metadata=metadata,
                indexes=indexes,
                lang='ru',
                batch_size=8)

result_texts, result_metadata = rag.retrieve(query="what county is dewitt michigan in?", k=15)
```
---
### LLM
Example of using LLM for your RAG.
```python
from ragify.model import LLMModel

model = LLMModel(model_name="Qwen/Qwen2-7B-Instruct")

system_prompt: str = ("You are a RAG assistant that answers the user's question based on the texts provided to you. "
                      "Formulate your answer in the same language as the texts. If there is no answer in the texts, refuse to answer.")

prompt = f"texts:\n{result_texts}question:{query}\nYour answer:"

response = model.generate(system_prompt=system_prompt, prompt=prompt)
```
---
### auto_preprocessing
The auto_preprocessing function automatically splits data into chunks and returns the texts. If metadata is present, the function also returns an array of indexes corresponding to each chunk along with the metadata.
> [!WARNING]
>For the function to work correctly, the data must follow the same format as in the example.
```python
from ragify.utils import auto_preprocessing
import numpy as np

dataset = {
    "data": [
        {
            "title": "sum of squares of even numbers formula",
            "context": "",
            #"metadata": "https://..."
        }
    ]
}

data = pd.DataFrame(dataset['data'])

corpus, indexes, metadata = auto_preprocessing(data)
```
---
### ContextualChunking
Here's an example of using contextual chunking.
```python
from ragify.utils import ContextualChunking

chunking = ContextualChunking(corpus=texts)

texts_with_context = chunking.run_contextual_chunking()
```
---
## 5. Installation :inbox_tray:
First, clone this GitHub repository:

```shell
git clone https://github.com/MoiseevArt/Ragify.git
```

Navigate to the `ragify` folder and install dependencies listed in `requirements.txt`. Easiest way is to use a virtual environment and install the dependencies.

```shell
cd ragify
python -m venv ../venv
```
Activate a virtual environment:
- For Windows:
```shell
..\venv\Scripts\activate
```

- For macOS/Linux:
```shell
source ../venv/bin/activate
```

Dependency Installation:
```shell
pip install -r requirements.txt
```
> [!WARNING]
>This project uses the CUDA version of PyTorch to run LLMs. PyTorch is not specified in `requirements.txt`, so you can install the necessary for you version from the official website: https://pytorch.org/get-started/locally/
---

## 6. License

This code repository is licensed under [the MIT License](https://github.com/MoiseevArt/Ragify/blob/main/LICENSE).

---
