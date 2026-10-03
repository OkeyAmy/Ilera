"""Create/update the Ilera-ATLAS demo Space on Hugging Face (free CPU basic hardware).

    pip install huggingface_hub
    python serve/deploy_space.py --user <you> --token hf_... [--modal-url https://...modal.run --vllm-key ...] [--log]

It copies serve/space/, replaces system_prompt.txt with the exact one used in training
(config/system_prompt.txt in <you>/ilera-data), and sets the Space secrets/variables.
The Space starts private; make it public on submission day.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SPACE_DIR = Path(__file__).parent / "space"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--space", default="ilera-atlas-demo")
    ap.add_argument("--modal-url")
    ap.add_argument("--vllm-key")
    ap.add_argument("--log", action="store_true", help="log anonymous interactions to <user>/ilera-data/logs")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()

    api = HfApi(token=a.token)
    repo = f"{a.user}/{a.space}"
    api.create_repo(repo, repo_type="space", space_sdk="gradio", private=not a.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(SPACE_DIR, tmp, dirs_exist_ok=True)
        try:
            shutil.copy(hf_hub_download(f"{a.user}/ilera-data", "config/system_prompt.txt", repo_type="dataset",
                                        token=a.token), Path(tmp) / "system_prompt.txt")
            print("system prompt: taken from ilera-data/config/system_prompt.txt")
        except Exception as e:
            print("WARNING: using the bundled system_prompt.txt:", repr(e)[:120])
        api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="space", commit_message="deploy Ilera-ATLAS demo",
                          ignore_patterns=["logs/*", "__pycache__/*"])

    api.add_space_secret(repo, "HF_TOKEN", a.token)
    api.add_space_variable(repo, "GGUF_REPO", f"{a.user}/ilera-atlas-GGUF")
    if a.modal_url:
        api.add_space_variable(repo, "MODAL_URL", a.modal_url)
    if a.vllm_key:
        api.add_space_secret(repo, "VLLM_API_KEY", a.vllm_key)
    if a.log:
        api.add_space_variable(repo, "LOG_REPO", f"{a.user}/ilera-data")
    print(f"https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
