"""Ilera-ATLAS chat UI, shared by serve/space/app.py and notebooks/07_demo.ipynb.

build_demo(respond) returns (demo, launch_kwargs). `respond(question)` is a generator that yields the
answer text as it grows (streaming) and finally yields the finished answer; the UI renders the final
answer as a colour-coded card. Call: demo.launch(**launch_kwargs, share=True).
"""
import inspect
import re

import gradio as gr

ACTION_RE = re.compile(r"ACTION:\s*\**\s*(TREAT|REFER|URGENT)", re.I)
BADGE = {
    "URGENT": ("🔴", "URGENT: refer immediately"),
    "REFER": ("🟠", "REFER to a health facility"),
    "TREAT": ("🟢", "TREAT at the primary health centre"),
}
NOTICE = "<sub>Decision support for trained health workers, following the 2024 National Standing Orders. Not a diagnosis.</sub>"

EXAMPLES = {
    "English": [
        "A 2-year-old has had fever for 3 days and is breathing fast. What should I do?",
        "A 6-month-old girl has had watery diarrhoea for 2 days and is still breastfeeding. What should I do?",
        "A pregnant woman at 32 weeks has a severe headache and blurred vision.",
        "A child was bitten by a dog this morning. What should I do?",
    ],
    "Hausa": [
        "Yaro mai shekara biyu yana da zazzabi kwana uku da saurin numfashi. Me zan yi?",
        "Mace mai ciki tana zubar da jini sosai. Me zan yi?",
    ],
    "Yoruba": [
        "Ọmọ ọdún kan ní ìgbẹ́ gbuuru fún ọjọ́ mẹ́ta. Kí ni kí n ṣe?",
        "Ọmọ oṣù mẹ́fà ní ibà àti ikọ́ fún ọjọ́ méjì. Kí ni kí n ṣe?",
    ],
    "Danger sign (safety layer)": [
        "A 1-year-old has convulsions and fever.",
        "A newborn is not feeding and is very sleepy.",
    ],
}

CSS = """
#ilera-header {background: linear-gradient(135deg, #0b6e4f 0%, #14946a 60%, #1fb382 100%); color: #fff;
  padding: 18px 22px; border-radius: 14px; margin-bottom: 6px;}
#ilera-header h1 {margin: 0; font-size: 1.55rem; color: #fff; letter-spacing: .2px;}
#ilera-header p {margin: 6px 0 0; opacity: .95; font-size: .95rem;}
#ilera-header .chips span {display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
  padding: 2px 10px; margin: 8px 6px 0 0; font-size: .8rem;}
.ilera-side h3 {margin-top: 4px;}
.ilera-example button {text-align: left !important; justify-content: flex-start !important; white-space: normal !important;}
.ilera-legend {font-size: .88rem; line-height: 1.6;}
"""

HEADER = """
<div id="ilera-header">
  <h1>🩺 Ilera-ATLAS</h1>
  <p>Primary-health-care decision support for <b>trained community health workers</b>, following
     Nigeria's <b>2024 National Standing Orders</b>.</p>
  <div class="chips"><span>English</span><span>Hausa</span><span>Yoruba</span>
     <span>N-ATLaS fine-tuned · Powered by Awarri</span></div>
</div>
"""

LEGEND = """
<div class="ilera-legend">

**How to read an answer**

🟢 **TREAT**: manage at the primary health centre<br>
🟠 **REFER**: send to a health facility<br>
🔴 **URGENT**: emergency, refer immediately<br>
📖 the Standing Orders section the answer comes from

⚠️ If a danger sign is mentioned, the answer is always switched to **URGENT**.

<sub>Not a diagnosis tool. Do not type patient names or personal details; questions may be logged
anonymously to improve the system.</sub>
</div>
"""


def render(answer, flag=None):
    """Turn 'ACTION / WHAT TO DO / DANGER SIGNS / SOURCE' text into a colour-coded markdown card."""
    m = ACTION_RE.search(answer)
    action = m.group(1).upper() if m else None
    steps = re.search(r"WHAT TO DO\s*:?\s*\n?([\s\S]*?)(?=\n\s*DANGER SIGNS|\n\s*SOURCE\s*:|\Z)", answer, re.I)
    danger = re.search(r"DANGER SIGNS\s*:\s*(.+)", answer, re.I)
    source = re.search(r"SOURCE\s*:\s*(.+)", answer, re.I)
    parts = []
    if flag:
        parts.append(f"> ⚠️ **Danger sign detected: _{flag}_.** Refer to a health facility now.")
    if action:
        icon, label = BADGE[action]
        parts.append(f"### {icon} {label}")
    if steps and steps.group(1).strip():
        lines = [l.strip() for l in steps.group(1).strip().splitlines() if l.strip()]
        parts.append("**What to do**\n\n" + "\n".join(lines))
    if danger and danger.group(1).strip().lower().rstrip(".") not in ("none listed", "none", "babu", ""):
        parts.append(f"**⚠️ Danger signs:** {danger.group(1).strip()}")
    if source:
        parts.append(f"📖 *{source.group(1).strip()}*")
    if not (action or (steps and steps.group(1).strip())):
        parts = ([parts[0]] if flag else []) + [answer]          # model ignored the format: show it as is
    parts.append(NOTICE)
    return "\n\n".join(parts)


def _supports(fn, name):
    return name in inspect.signature(fn).parameters


def build_demo(respond, title="Ilera-ATLAS"):
    """respond(question) -> generator of (text, final: bool, flag: str|None)."""
    blocks_kw = {"title": title}
    launch_kw = {}
    theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="green", neutral_hue="slate")
    for key, val in (("theme", theme), ("css", CSS)):
        (blocks_kw if _supports(gr.Blocks.__init__, key) else launch_kw)[key] = val   # Gradio 5 vs 6

    chat_kw = {"height": 560, "show_label": False, "render_markdown": True,
               "placeholder": "Ask about a patient, e.g. <i>A 2-year-old has fever and fast breathing…</i>"}
    if _supports(gr.Chatbot.__init__, "type"):
        chat_kw["type"] = "messages"
    if _supports(gr.Chatbot.__init__, "buttons"):
        chat_kw["buttons"] = ["copy"]
    elif _supports(gr.Chatbot.__init__, "show_copy_button"):
        chat_kw["show_copy_button"] = True

    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML(HEADER)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                chat = gr.Chatbot(**chat_kw)
                with gr.Row():
                    box = gr.Textbox(placeholder="Describe the patient (age, symptoms, how long) in English, Hausa or Yoruba",
                                     show_label=False, scale=8, autofocus=True, lines=1)
                    ask = gr.Button("Ask", variant="primary", scale=1)
                new = gr.Button("🗑 New conversation", size="sm")
            with gr.Column(scale=2, elem_classes="ilera-side"):
                gr.Markdown("### Try an example")
                for i, (group, questions) in enumerate(EXAMPLES.items()):
                    with gr.Accordion(group, open=(i == 0)):
                        for q in questions:
                            b = gr.Button(q, size="sm", elem_classes="ilera-example")
                            b.click(lambda q=q: q, None, box)
                gr.Markdown(LEGEND)

        def user_turn(message, history):
            message = (message or "").strip()
            if not message:
                return "", history
            return "", (history or []) + [{"role": "user", "content": message}]

        def bot_turn(history):
            if not history or history[-1]["role"] != "user":
                yield history
                return
            question = history[-1]["content"]
            history = history + [{"role": "assistant", "content": "⏳ *Reading the Standing Orders…*"}]
            yield history
            for text, final, flag in respond(question):
                history[-1]["content"] = render(text, flag) if final else text + " ▌"
                yield history

        for trigger in (box.submit, ask.click):
            trigger(user_turn, [box, chat], [box, chat], queue=False).then(bot_turn, chat, chat)
        new.click(lambda: [], None, chat, queue=False)
    return demo, launch_kw


if __name__ == "__main__":       # quick visual check without a model
    import time

    def fake(q):
        out = "ACTION: REFER\nWHAT TO DO:\n1. Give the first dose of amoxicillin.\n2. Refer to the health facility today.\nDANGER SIGNS: chest indrawing, unable to drink\nSOURCE: CHEW Standing Orders 2024, Section 2.5 Cough/Difficult Breathing, p. 59-63"
        for i in range(0, len(out), 25):
            time.sleep(0.05)
            yield out[:i], False, None
        yield out, True, None

    d, kw = build_demo(fake)
    d.queue().launch(**kw)
