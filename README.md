# LLM Fine-Tuning + Hybrid RAG with Mistral-7B for Scientific QA

This repository presents an end-to-end pipeline for fine-tuning and deploying a domain-specialized LLM for scientific PDF question answering. The backbone model, **Mistral-7B-Instruct-v0.3**, was fine-tuned using **qLoRA (4-bit LoRA adapters)** on papers related to LLM fine-tuning.

> **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/samyakshrestha/mistral-pdf-rag-app)  
> **Final Model**: [samyakshrestha/merged-finetuned-mistral](https://huggingface.co/samyakshrestha/merged-finetuned-mistral)

---

## Project Highlights

- **Curated 234 instruction-style QA pairs** from 24 research papers focused on LLM fine-tuning  
- **Formatted and tokenized the dataset** using Hugging Face `datasets` and `AutoTokenizer` for causal language modeling  
- **Fine-tuned Mistral-7B-Instruct-v0.3** using Hugging Face `transformers` and `peft` with **LoRA adapters** for parameter-efficient training  
- **Applied 4-bit quantization** via `BitsAndBytes` for low-memory training and inference  
- **Built a semantic RAG corpus** from 75 papers using token-aware chunking, `sentence-transformers` embeddings, and FAISS indexing  
- **Implemented hybrid RAG retrieval** with **LangChain**, combining context from both uploaded PDFs and a global semantic corpus  
- **Evaluated the model** using BLEU, BERTScore, and GPT-4o as an LLM-as-Judge  
- **Deployed the full pipeline** as a Dockerized Gradio app on Hugging Face Spaces with 4-bit quantized inference

---

## Dataset Overview

| Dataset Type     | Papers | QA Pairs / Chunks     |
|------------------|--------|-----------------------|
| Fine-Tuning QA   | 24     | **234** curated QAs   |
| RAG Corpus       | 75     | **~8k** text chunks   |
| Evaluation Set   | 3      | **30** unseen QAs     |

Documents were sourced from arXiv, focused on topics such as instruction tuning, LoRA, and prompt optimization.

---

## LoRA Fine-Tuning

- **Base model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Frameworks used**: `transformers`, `peft`, `datasets`
- **Quantization**: 4-bit using `BitsAndBytesConfig` (via `bitsandbytes`)
- **LoRA Adapters** saved to: `results/finetuned-mistral/` (LoRA weights only)
- **Full merged model** link:  
   [samyakshrestha/merged-finetuned-mistral](https://huggingface.co/samyakshrestha/merged-finetuned-mistral)

---

## Evaluation Results (Open Book)

The fine-tuned model outperformed the baseline in all of the following metrics:

| Metric               | Baseline (w/ Context)  | Fine-Tuned (w/ Context)   | Δ Improvement  |
|----------------------|------------------------|---------------------------|----------------|
| **BLEU-1**           | 0.2700                 | 0.4010                    | +0.1310        |
| **BLEU-4**           | 0.0855                 | 0.1625                    | +0.0770        |
| **BERTScore (F1)**   | 0.4060                 | 0.5295                    | +0.1235        |
| **LLM-as-Judge**     | 3.73 / 5               | 3.97 / 5                  | +0.24          |


--- 

## RAG Corpus Preparation (Chunk → Embed → FAISS Index)

In the notebook (`13_chunk_and_embed.ipynb`), we constructed a semantic retrieval backend for the RAG system by transforming 75 scientific papers into a searchable FAISS database.

### Chunking Strategy
- Token-based sliding window (~256 tokens per chunk, stride of 50) for robust segmentation.
- Preserves semantic continuity across chunk boundaries.
- Output: **8,724 semantic chunks** across 75 papers.

### Embedding
- Model: `BAAI/bge-base-en-v1.5` (via `sentence-transformers`)
- Output: 768-dimensional L2-normalized vectors.
- Stored in: `chunk_embeddings.npy`
- Metadata stored in: `chunk_metadata.json`

### Indexing
- Index Type: `faiss.IndexFlatIP` (Inner Product for cosine similarity)
- Fast, scalable, non-quantized retrieval
- Output: `faiss_index.bin`

---

## Deployment & Hybrid RAG Architecture

The final application was deployed on **Hugging Face Spaces** using a **Dockerized Gradio interface**, allowing users to upload scientific PDFs and query them via a lightweight semantic QA system.

### Hybrid RAG Retrieval Logic

The system performs dual-source retrieval, combining local and global context:

- **PDF Vectorstore**  
  Upon upload, the PDF is parsed, chunked using a 256-token sliding window, and embedded in-memory using `BAAI/bge-base-en-v1.5`.

- **Corpus Vectorstore**  
  A global semantic index (~8.7K chunks across 75 LLM fine-tuning papers) is precomputed and downloaded at runtime from [Hugging Face Hub](https://huggingface.co/samyakshrestha/merged-finetuned-mistral).

- **Retrieval Pipeline**  
  - Retrieve top-4 chunks from the **PDF** vectorstore  
  - Retrieve top-2 chunks from the **global FAISS** corpus  
  - Concatenate results into a unified prompt with `[PDF CONTEXT]` and `[CORPUS CONTEXT]` labels  
  - Pass the prompt to the fine-tuned **Mistral-7B** model for generation

### Deployment Stack

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Frontend**      | Gradio Blocks UI (file upload + QA textbox)                                |
| **Model**         | Mistral-7B-Instruct v0.3 (LoRA adapters merged)                             |
| **Quantization**  | 4-bit inference via `BitsAndBytes` (`bnb_config`)                          |
| **Backend**       | Python app (`gradio_app.py`) containerized via `Dockerfile`                |
| **Storage**       | FAISS index + chunk metadata downloaded from `huggingface_hub` on startup  |
| **Runtime**       | Hosted on Hugging Face T4 GPU instance                                     |

> [**Live Demo**](https://huggingface.co/spaces/samyakshrestha/mistral-pdf-rag-app)

---

## Technology Stack

- **LLM Stack**: **Transformers**, **PEFT**, **BitsAndBytes**, **Datasets**, **SentenceTransformers**
- **RAG Stack**: **FAISS**, **HuggingFace Hub**, **LangChain**
- **Deployment**: **Docker**, **Gradio**, **FastAPI**, **Hugging Face Spaces**
- **Evaluation**: **BLEU**, **BERTScore**, **GPT-4 (LLM-as-Judge)**