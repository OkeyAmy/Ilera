"""Smoke-test the Ilera-ATLAS API (Modal vLLM endpoint).

    python serve/test_api.py --url https://<...>.modal.run --key <VLLM_API_KEY>

The first call after the endpoint has been idle can take 1-3 minutes (GPU start + model load).
"""
import argparse
import re
import time

from openai import OpenAI

QUESTIONS = [
    "A 2-year-old has had fever for 3 days and is breathing fast. What should I do?",
    "Yaro mai shekara biyu yana da zazzabi da saurin numfashi. Me zan yi?",
    "Ọmọ ọdún kan ní ìgbẹ́ gbuuru fún ọjọ́ mẹ́ta. Kí ni kí n ṣe?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Modal URL, without /v1")
    ap.add_argument("--key", required=True, help="VLLM_API_KEY")
    ap.add_argument("--system", default="serve/space/system_prompt.txt")
    a = ap.parse_args()
    system = open(a.system, encoding="utf-8").read().strip()
    client = OpenAI(base_url=a.url.rstrip("/") + "/v1", api_key=a.key, timeout=300)
    for q in QUESTIONS:
        t = time.time()
        r = client.chat.completions.create(
            model="ilera-atlas", temperature=0, max_tokens=450,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": q}],
            extra_body={"repetition_penalty": 1.1},
        )
        text = r.choices[0].message.content
        ok = bool(re.search(r"ACTION:\s*(TREAT|REFER|URGENT)", text))
        print(f"{'OK ' if ok else 'BAD'} {time.time() - t:5.1f}s  {q}\n{text}\n")


if __name__ == "__main__":
    main()
