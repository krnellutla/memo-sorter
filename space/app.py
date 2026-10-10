import gradio as gr
from transformers import pipeline

clf = pipeline("text-classification", model="knellutla/memo-sorter-minilm", top_k=None)

EXAMPLES = [
    "Dentist, Thursday 9:30",
    "A browser extension that rewrites error messages in plain English",
    "Replace the batteries in the smoke alarms",
    "Rainy morning. Made coffee and felt oddly peaceful",
]


def sort_note(text):
    if not text.strip():
        return {}
    return {r["label"]: float(r["score"]) for r in clf(text[:500])[0]}


demo = gr.Interface(
    fn=sort_note,
    inputs=gr.Textbox(label="Your note", lines=3, placeholder="Type a short note..."),
    outputs=gr.Label(num_top_classes=4, label="Tag"),
    examples=EXAMPLES,
    title="Memo Sorter",
    description="A 22M-parameter model that tags a note as idea, todo, reminder or journal. "
                "It scored 83% on 48 unseen notes, and it still mixes up ideas and todos. "
                "Part of tinkeryard.dev.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
