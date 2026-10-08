"""Ilera-ATLAS demo + API on a free Hugging Face ZeroGPU Space.

Free personal accounts (older than 30 days) can host up to 2 ZeroGPU Spaces, so this is a free, permanent
link. The merged model loads once; each answer borrows a shared GPU through @spaces.GPU. A red-flag
safety layer forces an urgent referral whenever a danger sign is mentioned, whatever the model says.

Space secrets / variables:
    HF_TOKEN    read access to the private model repo (and LOG_REPO if used)
    MODEL_REPO  optional, default Emmanuel-okoye/ilera-atlas (merged 16-bit model)
    LOG_REPO    optional dataset repo for anonymous interaction logs, e.g. <you>/ilera-data
"""
import csv
import datetime
import json
import os
import re
import uuid
from pathlib import Path
from threading import Thread

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import ui

HERE = Path(__file__).parent
SYSTEM = (HERE / "system_prompt.txt").read_text(encoding="utf-8").strip()
MODEL_REPO = os.getenv("MODEL_REPO", "Emmanuel-okoye/ilera-atlas")
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", 450))


def load_flags(path=HERE / "redflags.csv"):
    """Danger-sign phrases in every language column, lower-cased."""
    flags = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flags += [v.strip().lower() for v in row.values() if v and v.strip()]
    return sorted(set(flags), key=len, reverse=True)


FLAGS = load_flags()


def red_flag(question):
    q = " " + re.sub(r"\s+", " ", question.lower()) + " "
    return next((f for f in FLAGS if f in q), None)


def force_urgent(answer):
    if ui.ACTION_RE.search(answer):
        return ui.ACTION_RE.sub("ACTION: URGENT", answer, count=1)
    return "ACTION: URGENT\n" + answer


# ---- model: loaded once; ZeroGPU emulates CUDA outside @spaces.GPU -----------
tok = AutoTokenizer.from_pretrained(MODEL_REPO, token=os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, token=os.getenv("HF_TOKEN"), dtype=torch.bfloat16).to("cuda")
model.generation_config.max_length = None


@spaces.GPU(duration=60)
def stream(question):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs=dict(**enc, streamer=streamer, max_new_tokens=MAX_NEW, do_sample=False,
                                              repetition_penalty=1.1,
                                              pad_token_id=tok.pad_token_id or tok.eos_token_id)).start()
    out = ""
    for piece in streamer:
        out += piece
        yield out


# ---- optional anonymous logging (for the user test) -------------------------
_scheduler = None
LOG_DIR = HERE / "logs"
if os.getenv("LOG_REPO"):
    from huggingface_hub import CommitScheduler

    LOG_DIR.mkdir(exist_ok=True)
    _scheduler = CommitScheduler(repo_id=os.environ["LOG_REPO"], repo_type="dataset", folder_path=LOG_DIR,
                                 path_in_repo="logs", every=5, token=os.getenv("HF_TOKEN"))
LOG_FILE = LOG_DIR / f"interactions-{uuid.uuid4().hex[:8]}.jsonl"


def log(question, answer, flag):
    if _scheduler is None:
        return
    with _scheduler.lock, open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.datetime.utcnow().isoformat(timespec="seconds"), "question": question,
                            "answer": answer, "red_flag": flag}, ensure_ascii=False) + "\n")


def respond(question):
    """For ui.build_demo: streamed text, then the finished answer with the safety layer applied."""
    flag = red_flag(question)
    text = ""
    for text in stream(question):
        yield text, False, None
    final = force_urgent(text) if flag else text
    log(question, final, flag)
    yield final, True, flag


def answer(question: str) -> str:
    """API: one question in, the final answer (plain text, safety layer applied) out."""
    question = (question or "").strip()
    if not question:
        return "Please type a question."
    final, flag = "", None
    for final, _, flag in respond(question):
        pass
    if flag:
        final = f"Danger sign detected ({flag}): refer to a health facility now.\n\n" + final
    return final + "\n\nDecision support for trained health workers. Not a diagnosis."


demo, launch_kw = ui.build_demo(respond)
with demo:
    gr.api(answer, api_name="answer")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(**launch_kw)
