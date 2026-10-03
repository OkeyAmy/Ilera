"""Ilera-ATLAS GPU API on Modal (runbook Part 11.1).

OpenAI-compatible endpoint (vLLM) serving the merged model on one L4 GPU. Scales to zero when idle,
so it only spends Modal's free monthly credits while answering.

One-time setup on your laptop:
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_...            # token that can read the private model
    modal secret create ilera-api VLLM_API_KEY=<long-random>   # clients must send this as Bearer token

Deploy (prints the URL):
    MODEL_REPO=<you>/ilera-atlas modal deploy serve/modal_app.py

Test:
    python serve/test_api.py --url https://<...>.modal.run --key <VLLM_API_KEY>
"""
import os

import modal

MODEL_REPO = os.environ.get("MODEL_REPO", "YOUR_HF_USERNAME/ilera-atlas")
SERVED_NAME = "ilera-atlas"
MINUTES = 60
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "MODEL_REPO": MODEL_REPO})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("ilera-atlas")


@app.server(
    image=image,
    gpu="L4",                               # 24 GB: holds the 16 GB fp16 model; ~$0.80/h only while running
    scaledown_window=5 * MINUTES,           # back to zero (no cost) 5 min after the last request
    startup_timeout=15 * MINUTES,           # first start downloads ~16 GB into the cache volume
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("ilera-api")],
    port=PORT,
    target_concurrency=16,
    unauthenticated=True,                   # Modal lets requests through; vLLM checks VLLM_API_KEY
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm", "serve", os.environ["MODEL_REPO"],
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0", "--port", str(PORT),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.92",
            "--enforce-eager",              # faster cold start, which matters with scale-to-zero
        ]
        print(*cmd)
        self.process = subprocess.Popen(cmd)   # VLLM_API_KEY comes from the "ilera-api" secret

    @modal.exit()
    def stop(self):
        self.process.terminate()
