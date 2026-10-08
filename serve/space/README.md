---
title: Ilera-ATLAS
emoji: 🩺
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: N-ATLaS primary-health assistant for Nigerian CHEWs
---

# Ilera-ATLAS demo (Powered by Awarri)

N-ATLaS fine-tuned to follow Nigeria's 2024 National Standing Orders for community health workers,
in English, Hausa and Yoruba. Decision support for trained health workers only; not a diagnosis tool.

Runs on a free ZeroGPU Space. Each visitor gets a free daily GPU allowance from Hugging Face
(more when signed in). A red-flag safety layer forces an urgent referral when a danger sign is mentioned.

API: `gradio_client.Client("Emmanuel-okoye/ilera-atlas-demo").predict("<question>", api_name="/answer")`.

Derived from N-ATLaS under the N-ATLaS Terms of Use. Attribution: Awarri Technologies and the Federal
Ministry of Communications, Innovation and Digital Economy.
