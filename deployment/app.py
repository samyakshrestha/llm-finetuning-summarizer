# app.py – Finetuned-Mistral RAG (FastAPI + Gradio)
import os, json, warnings, logging
from pathlib import Path
from typing import List

import torch, faiss
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import hf_hub_download

# Silence noisy logs
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

# Configuration
HF_MODEL_ID   = "samyakshrestha/merged-finetuned-mistral"
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K         = 5
CTX_TOKEN_LIMIT = 2048
MAX_NEW_TOKENS  = 256

# Cache dirs
DATA_DIR   = Path("/data");            DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR  = DATA_DIR / "hf_cache";    CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)

# Files inside model repo
FAISS_BIN_REPO = "data/faiss_index/faiss_index.bin"
META_JSON_REPO = "data/faiss_index/chunk_metadata.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
META_PATH  = DATA_DIR / "chunk_metadata.json"

# 1. Embeddings
print("Loading embedder …")
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
print(f"{EMBED_MODEL} ({embedder.get_sentence_embedding_dimension()}-d)")

# 2. FAISS index
def fetch_faiss():
    idx  = hf_hub_download(HF_MODEL_ID, FAISS_BIN_REPO, local_dir=DATA_DIR, local_dir_use_symlinks=False)
    meta = hf_hub_download(HF_MODEL_ID, META_JSON_REPO, local_dir=DATA_DIR, local_dir_use_symlinks=False)
    return Path(idx), Path(meta)

if not INDEX_PATH.exists() or not META_PATH.exists():
    INDEX_PATH, META_PATH = fetch_faiss()

print("Loading FAISS …")
index = faiss.read_index(str(INDEX_PATH))
with META_PATH.open() as f:
    chunk_metadata = json.load(f)
print(f"vectors = {index.ntotal}")

# 3. Load model (GPU with 4-bit quant)
print("Loading Mistral-7B …")
bnb_cfg = None
if torch.cuda.is_available():
    try:
        import bitsandbytes, accelerate  # check they exist
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("   4-bit quant (GPU)")
    except Exception as e:
        print(f"   4-bit fallback → fp16 ({e})")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)
model.eval()
print("   model ready")

# 4. RAG helpers
def retrieve_chunks(q: str, k: int = TOP_K) -> List[dict]:
    emb = embedder.encode([q], normalize_embeddings=True)
    _, idx = index.search(emb, k)
    return [chunk_metadata[int(i)] for i in idx[0]]

def build_prompt(q: str, chunks: List[dict]) -> str:
    ctx, tokens = [], 0
    for ch in chunks:
        block = f"[{ch['title']}]\n{ch['text']}\n"
        t     = len(tokenizer.tokenize(block))
        if tokens + t <= CTX_TOKEN_LIMIT:
            ctx.append(block)
            tokens += t
    return (
        "You are an expert scientific assistant. Use the excerpts to answer.\n\n"
        + "Excerpts:\n" + "\n\n".join(ctx) +
        f"\n\nQuestion: {q}\nAnswer:"
    )

@torch.inference_mode()
def generate_answer(q: str) -> str:
    prompt = build_prompt(q, retrieve_chunks(q))
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()

# 5. FastAPI backend
api = FastAPI(title="Finetuned-Mistral RAG")
class Question(BaseModel): question: str
class Answer(BaseModel): answer: str

@api.post("/rag", response_model=Answer)
def rag(item: Question):
    return Answer(answer=generate_answer(item.question))

# 6. Gradio UI
demo = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(label="Ask a question about LLM fine-tuning"),
    outputs=gr.Textbox(label="Answer"),
    title="Finetuned Mistral-7B • RAG demo",
    allow_flagging="never"  # this disables the 'flag' folder that caused the crash
)

# 7. Launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)