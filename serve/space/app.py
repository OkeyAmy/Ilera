"""Ilera-ATLAS demo + API on a free Hugging Face Space (runbook Part 11.2).

Answers come from the Modal GPU API when MODAL_URL is set and reachable; otherwise from the GGUF
model on the Space's own CPU (slow, but always available). A red-flag safety layer forces an urgent
referral whenever a danger sign is mentioned, whatever the model says.

Space secrets / variables:
    HF_TOKEN       read access to the private model repos (and LOG_REPO if used)
    GGUF_REPO      e.g. <you>/ilera-atlas-GGUF
    MODAL_URL      optional, e.g. https://<...>.modal.run   (without /v1)
    VLLM_API_KEY   optional, the key set on Modal
    LOG_REPO       optional dataset repo for anonymous interaction logs, e.g. <you>/ilera-data
"""
import csv
import datetime
import json
import os
import re
import uuid
from pathlib import Path

import gradio as gr

HERE = Path(__file__).parent
SYSTEM = (HERE / "system_prompt.txt").read_text(encoding="utf-8").strip()
NOTICE = "\n\n_Decision support for trained health workers. Not a diagnosis._"
ACTION_RE = re.compile(r"ACTION:\s*(TREAT|REFER|URGENT)", re.I)


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


def apply_safety(question, answer):
    """Force URGENT when a danger sign is in the question; add the notice."""
    hit = red_flag(question)
    if hit:
        if ACTION_RE.search(answer):
            answer = ACTION_RE.sub("ACTION: URGENT", answer, count=1)
        else:
            answer = "ACTION: URGENT\n" + answer
        answer = f"⚠ Danger sign detected ({hit}): refer to a health facility now.\n\n" + answer
    return answer + NOTICE


# ---- model backends --------------------------------------------------------
_remote = None
if os.getenv("MODAL_URL"):
    from openai import OpenAI

    _remote = OpenAI(base_url=os.environ["MODAL_URL"].rstrip("/") + "/v1",
                     api_key=os.getenv("VLLM_API_KEY", "none"), timeout=180)
_local = None


def local_llm():
    global _local
    if _local is None:
        from llama_cpp import Llama

        _local = Llama.from_pretrained(os.environ["GGUF_REPO"], filename=os.getenv("GGUF_FILE", "*Q4_K_M*.gguf"),
                                       n_ctx=2048, n_threads=os.cpu_count() or 2, verbose=False)
    return _local


def generate(question):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    if _remote is not None:
        try:
            r = _remote.chat.completions.create(model="ilera-atlas", messages=msgs, temperature=0,
                                                max_tokens=450, extra_body={"repetition_penalty": 1.1})
            return r.choices[0].message.content, "gpu"
        except Exception as e:  # GPU API down or cold-starting too long: use the CPU copy
            print("remote failed, using local GGUF:", repr(e)[:200])
    out = local_llm().create_chat_completion(messages=msgs, temperature=0, max_tokens=int(os.getenv("CPU_MAX_TOKENS", 300)),
                                             repeat_penalty=1.1)
    return out["choices"][0]["message"]["content"], "cpu"


# ---- optional anonymous logging (for the user test) -------------------------
_scheduler = None
LOG_DIR = HERE / "logs"
if os.getenv("LOG_REPO"):
    from huggingface_hub import CommitScheduler

    LOG_DIR.mkdir(exist_ok=True)
    _scheduler = CommitScheduler(repo_id=os.environ["LOG_REPO"], repo_type="dataset", folder_path=LOG_DIR,
                                 path_in_repo="logs", every=5, token=os.getenv("HF_TOKEN"))
LOG_FILE = LOG_DIR / f"interactions-{uuid.uuid4().hex[:8]}.jsonl"


def log(question, answer, backend):
    if _scheduler is None:
        return
    with _scheduler.lock, open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                            "question": question, "answer": answer, "backend": backend}, ensure_ascii=False) + "\n")


def answer(message, history=None):
    question = (message or "").strip()
    if not question:
        return "Please type a question."
    text, backend = generate(question)
    final = apply_safety(question, text)
    log(question, final, backend)
    return final


DESCRIPTION = """Primary-health-care decision support for **trained Nigerian community health workers**, following the
**2024 National Standing Orders**. Ask in **English, Hausa or Yoruba**. N-ATLaS fine-tuned · Powered by Awarri.

Not a diagnosis tool. Do not type patient names or other personal details. Questions may be logged
anonymously to improve the system."""

demo = gr.ChatInterface(
    answer,
    title="Ilera-ATLAS",
    description=DESCRIPTION,
    examples=["A 2-year-old has had fever for 3 days and is breathing fast. What should I do?",
              "A pregnant woman has a severe headache and blurred vision.",
              "Yaro mai shekara biyu yana da zazzabi da saurin numfashi. Me zan yi?",
              "Ọmọ ọdún kan ní ìgbẹ́ gbuuru fún ọjọ́ mẹ́ta. Kí ni kí n ṣe?"],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
