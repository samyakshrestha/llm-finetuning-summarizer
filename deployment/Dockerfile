FROM python:3.10-slim

# System deps for git-lfs (model pulls) and faster tokenization wheels
RUN apt-get update && apt-get install -y git-lfs && git lfs install
RUN mkdir -p /data && chmod 777 /data
ENV TRANSFORMERS_CACHE=/data/hf_cache

WORKDIR /app

# Install Python deps first for cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
# Sentence-Transformer
snapshot_download("BAAI/bge-base-en-v1.5")
# LoRA-merged Mistral
snapshot_download("samyakshrestha/merged-finetuned-mistral")
# FAISS index & metadata
hf_hub_download("samyakshrestha/merged-finetuned-mistral",
                "data/faiss_index/faiss_index.bin")
hf_hub_download("samyakshrestha/merged-finetuned-mistral",
                "data/faiss_index/chunk_metadata.json")
PY

# Copy application code
COPY . .

EXPOSE 7860
CMD ["python", "app.py"]