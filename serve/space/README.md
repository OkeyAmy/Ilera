---
title: Ilera-ATLAS
emoji: 🩺
colorFrom: green
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: other
short_description: N-ATLaS primary-health assistant for Nigerian CHEWs
---

# Ilera-ATLAS demo (Powered by Awarri)

N-ATLaS fine-tuned to follow Nigeria's 2024 National Standing Orders for community health workers,
in English, Hausa and Yoruba. Decision support for trained health workers only; not a diagnosis tool.

Answers come from a GPU API (Modal, vLLM) when available, otherwise from a quantised copy running on
this Space's CPU. A red-flag safety layer forces an urgent referral when a danger sign is mentioned.

Derived from N-ATLaS under the N-ATLaS Terms of Use. Attribution: Awarri Technologies and the Federal
Ministry of Communications, Innovation and Digital Economy.
