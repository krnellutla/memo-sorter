import gradio as gr
from transformers import pipeline

clf = pipeline("text-classification", model="knellutla/memo-sorter-minilm", top_k=None)

EXAMPLES = [
    "Dentist, Thursday 9:30",
    "A browser extension that rewrites error messages in plain English",
    "Replace the batteries in the smoke alarms",
    "Rainy morning. Made coffee and felt oddly peaceful",
]

# Colors and fonts match tinkeryard.dev
FLAG = gr.themes.Color(
    c50="#FFF0F4", c100="#FFDCE5", c200="#FFB9CB", c300="#FA8FAB", c400="#F4668B",
    c500="#EE3F6C", c600="#D62F5B", c700="#B22349", c800="#8C1C3B", c900="#6A152D", c950="#470E1F",
)
INK = gr.themes.Color(
    c50="#F2F3EC", c100="#E6E9E1", c200="#D9DDD4", c300="#BFC6BC", c400="#9AA39A",
    c500="#5C6A62", c600="#44524A", c700="#2F3D36", c800="#1F2D28", c900="#16231F", c950="#0E1714",
)
theme = gr.themes.Base(
    primary_hue=FLAG,
    secondary_hue=INK,
    neutral_hue=INK,
    font=[gr.themes.GoogleFont("Source Serif 4"), "Georgia", "serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#F2F3EC",
    body_background_fill_dark="#F2F3EC",
    body_text_color="#16231F",
    body_text_color_dark="#16231F",
    block_background_fill="#F2F3EC",
    block_background_fill_dark="#F2F3EC",
    block_border_width="0px",
    block_shadow="none",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#5C6A62",
    block_label_text_color_dark="#5C6A62",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#16231F",
    input_border_color_dark="#16231F",
    input_border_width="2px",
    button_primary_background_fill="#EE3F6C",
    button_primary_background_fill_hover="#D62F5B",
    button_primary_background_fill_dark="#EE3F6C",
    button_primary_background_fill_hover_dark="#D62F5B",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_dark="transparent",
    button_secondary_text_color="#16231F",
    button_secondary_text_color_dark="#16231F",
    button_secondary_border_color="#16231F",
    button_secondary_border_color_dark="#16231F",
    slider_color="#EE3F6C",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,600..800&display=swap');
.gradio-container{max-width:860px!important;background:#F2F3EC!important}
h1,h2,button,.label-wrap span,.confidence-set .label{font-family:'Bricolage Grotesque','Arial Narrow',sans-serif!important}
h1{font-stretch:75%;font-weight:800!important;font-size:3rem!important;letter-spacing:-.02em;line-height:.95!important;margin:0 0 .4rem!important}
button.primary,button.secondary{font-stretch:85%;font-weight:700!important;border-radius:2px!important}
button.secondary{border:2px solid #16231F!important}
.output-class{font-family:'Bricolage Grotesque',sans-serif!important;font-stretch:80%;font-weight:800!important;color:#EE3F6C!important}
.bar{background:#EE3F6C!important}
footer{display:none!important}
"""


def sort_note(text):
    if not text.strip():
        return {}
    return {r["label"]: float(r["score"]) for r in clf(text[:500])[0]}


with gr.Blocks(title="Memo Sorter") as demo:
    gr.Markdown(
        "# Memo Sorter\n"
        "A 22M-parameter model that tags a note as idea, todo, reminder or journal. "
        "It scored 83% on 48 unseen notes, and it still mixes up ideas and todos."
    )
    with gr.Row():
        with gr.Column():
            note = gr.Textbox(label="Your note", lines=4, placeholder="Type a short note...")
            with gr.Row():
                clear = gr.ClearButton(value="Clear")
                go = gr.Button("Sort it", variant="primary")
        out = gr.Label(num_top_classes=4, label="Tag")
    clear.add([note, out])
    gr.Examples(EXAMPLES, inputs=note, label="Try one of these")
    go.click(sort_note, note, out, api_name="sort_note")
    note.submit(sort_note, note, out, api_name=False)

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
