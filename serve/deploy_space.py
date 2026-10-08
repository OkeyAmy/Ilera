"""Create/update the Ilera-ATLAS demo Space on Hugging Face (free ZeroGPU hardware).

    pip install huggingface_hub
    python serve/deploy_space.py --user <you> --token hf_... [--log]

It copies serve/space/, replaces system_prompt.txt with the exact one used in training
(config/system_prompt.txt in <you>/ilera-data), sets the HF_TOKEN secret and requests ZeroGPU.
Free accounts older than 30 days can host 2 ZeroGPU Spaces. The Space is public (the model repo
can stay private: the Space reads it with the HF_TOKEN secret).
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
    ap.add_argument("--log", action="store_true", help="log anonymous interactions to <user>/ilera-data/logs")
    a = ap.parse_args()

    api = HfApi(token=a.token)
    repo = f"{a.user}/{a.space}"
    api.create_repo(repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
    api.add_space_secret(repo, "HF_TOKEN", a.token)
    api.add_space_variable(repo, "MODEL_REPO", f"{a.user}/ilera-atlas")
    if a.log:
        api.add_space_variable(repo, "LOG_REPO", f"{a.user}/ilera-data")

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
    print(f"https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
