# Ilera-ATLAS API

Two ways to call Ilera-ATLAS. Both use N-ATLaS fine-tuned on Nigeria's 2024 National Standing Orders.

| Endpoint | Speed | Safety layer | Auth |
|---|---|---|---|
| **GPU API** (Modal, vLLM, OpenAI-compatible) | fast (first call after idle: 1–3 min cold start) | no: raw model | `Authorization: Bearer <API key>` |
| **Demo Space API** (Gradio) | slower on CPU fallback | **yes**: red-flag override + notice | Hugging Face token while the Space is private |

## 0. Live API (free, always on) — try this first

The public demo runs Ilera-ATLAS (Q4_K_M GGUF, llama.cpp) on a free Kaggle CPU behind a Gradio app, with the
red-flag safety layer. The session restarts itself every ~11.5 h with a new Gradio URL, which is always published at
`https://huggingface.co/spaces/Emmanuel-okoye/ilera-atlas/resolve/main/link.txt`. Answers take 30–100 s on CPU.

```python
# pip install gradio_client
import urllib.request
from gradio_client import Client

url = urllib.request.urlopen("https://huggingface.co/spaces/Emmanuel-okoye/ilera-atlas/resolve/main/link.txt").read().decode().split()[0]
client = Client(url, httpx_kwargs={"timeout": 300})
_, history = client.predict("A pregnant woman at 32 weeks has a severe headache and blurred vision.", [], api_name="/user_turn")
history = client.predict(history, api_name="/bot_turn")
print(history[-1]["content"][0]["text"])     # Markdown answer card: action, steps, danger signs, source
```

Endpoints: `/user_turn(message, history) -> ("", history)` adds the question; `/bot_turn(history) -> history` appends the
answer (safety layer applied). The page's own "Use via API" link lists them too.

## Answer format

Every answer follows this contract (labels always in English, content in the question's language):

```
ACTION: TREAT | REFER | URGENT
WHAT TO DO: numbered steps
DANGER SIGNS: signs that mean urgent referral
SOURCE: document and section
```

Always send the system prompt from [`serve/space/system_prompt.txt`](../serve/space/system_prompt.txt);
without it the model does not use the format. Use `temperature=0`.

## 1. GPU API (OpenAI-compatible)

**curl**

```bash
curl https://<your-modal-url>/v1/chat/completions \
  -H "Authorization: Bearer $ILERA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "ilera-atlas", "temperature": 0, "max_tokens": 450, "repetition_penalty": 1.1,
       "messages": [{"role": "system", "content": "<system prompt>"},
                    {"role": "user", "content": "A 2-year-old has fever and fast breathing. What should I do?"}]}'
```

**Python**

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-modal-url>/v1", api_key="<API key>")
system = open("serve/space/system_prompt.txt", encoding="utf-8").read()
r = client.chat.completions.create(
    model="ilera-atlas", temperature=0, max_tokens=450,
    messages=[{"role": "system", "content": system},
              {"role": "user", "content": "Mace mai ciki tana zubar da jini sosai."}],
    extra_body={"repetition_penalty": 1.1})
print(r.choices[0].message.content)
```

Parse the action with `re.search(r"ACTION:\s*(TREAT|REFER|URGENT)", text)`.

## 2. Demo Space API (with the safety layer)

```python
from gradio_client import Client

c = Client("<you>/ilera-atlas-demo", hf_token="hf_...")   # hf_token only needed while the Space is private
print(c.predict("A pregnant woman has a severe headache and blurred vision.", api_name="/answer"))
```

## Limits and responsible use

* Decision support for **trained health workers**; not for patients; not a diagnosis.
* N-ATLaS licence: at most **1,000 active end-users per 30 days** without a commercial licence from
  Awarri / FMCIDE. Attribution: *Awarri Technologies and the Federal Ministry of Communications,
  Innovation and Digital Economy*. Powered by Awarri.
* Do not send personal or identifying patient data.
