# ------------------------
# Step 1: Import Libraries
# ------------------------

import os
import torch
import faiss
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# ------------------------
# Step 2: Load FAISS Index and Metadata
# ------------------------

# Set paths
faiss_index_path = "./data/rag_corpus/faiss_index.bin"
metadata_path = "./data/rag_corpus/chunk_metadata.json"

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load metadata
with open(metadata_path, "r") as f:
    chunk_metadata = json.load(f)

# ------------------------
# Step 3: Load Fine-Tuned Model and Tokenizer
# ------------------------

model_path = "./models/merged-finetuned-mistral"
#bnb_config = BitsAndBytesConfig(
    #load_in_4bit=True,
    #bnb_4bit_compute_dtype=torch.float16,
    #bnb_4bit_use_double_quant=True,
    #bnb_4bit_quant_type="nf4",
#)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

# Important: ensure padding token is set correctly
tokenizer.pad_token = tokenizer.eos_token

model.eval() # Ensure inference-only mode

# ------------------------
# Step 4: Semantic Retriever Function
# ------------------------

# Load embedding model for query encoding
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def retrieve_top_k(query, k=5):
    """Embed the query and retrieve top-k similar chunks from FAISS index."""
    query_emb = embedding_model.encode(query, normalize_embeddings=True)
    query_emb = query_emb.reshape(1, -1)  # FAISS expects (batch_size, dimension)
    distances, indices = index.search(query_emb, k)
    retrieved_chunks = [chunk_metadata[i] for i in indices[0]]
    return retrieved_chunks

# ------------------------
# Step 5: Prompt Construction
# ------------------------

def build_prompt(query, retrieved_chunks):
    """Construct prompt by combining context and question."""
    context_text = "\n\n".join(chunk['text'] for chunk in retrieved_chunks)
    prompt = (
        f"You are an expert scientific assistant. Use the provided excerpts to answer the question.\n\n"
        f"Excerpts:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt

# ------------------------
# Step 6: Final RAG Inference Function
# ------------------------

def rag_inference(query: str, 
                  top_k: int = 5, 
                  max_new_tokens: int = 512, 
                  max_length: int = 4096) -> str:
    """
    Full RAG pipeline: retrieve relevant chunks → build prompt → generate answer.

    Args:
        query (str): User's question.
        top_k (int): Number of chunks to retrieve from FAISS.
        max_new_tokens (int): Maximum number of tokens model can generate.
        max_length (int): Maximum total length of tokenized prompt.

    Returns:
        str: Model's generated answer.
    """
    
    # Retrieve top-k relevant chunks
    retrieved_chunks = retrieve_top_k(query, k=top_k)

    # Build the full prompt
    prompt = build_prompt(query, retrieved_chunks)

    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,    # Greedy decoding
            temperature=0.0,    # Make sure generation is deterministic
            top_p=1.0
        )

    # Decode and clean the generated answer
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract portion after "Answer:" to keep it clean
    if "Answer:" in decoded_output:
        final_answer = decoded_output.split("Answer:")[-1].strip()
    else:
        final_answer = decoded_output.strip()

    return final_answer