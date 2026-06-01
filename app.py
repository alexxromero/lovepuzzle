import gradio as gr

from clue_generator import load_model, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from puzzle import generate_puzzle, _validate_phone

print(f"Loading generator ({MODEL_ID})...")
g_model, g_tokenizer = load_model()

print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
v_model, v_tokenizer = load_verifier()


def run(phone_raw: str, domain1: str, domain2: str, domain3: str) -> str:
    phone_raw = phone_raw.strip()
    domain1, domain2, domain3 = domain1.strip(), domain2.strip(), domain3.strip()

    if not phone_raw:
        return "Please enter a phone number."
    domains = [d for d in [domain1, domain2, domain3] if d]
    if len(domains) < 3:
        return "Please enter all three domains."

    try:
        phone = _validate_phone(phone_raw)
    except ValueError as e:
        return f"Invalid phone number: {e}"

    try:
        eq, puzzle, _, _ = generate_puzzle(
            phone, domains, g_model, g_tokenizer, v_model, v_tokenizer
        )
    except ValueError as e:
        return str(e)

    return f"Equation: {eq['infix']}\n\n{puzzle}"


with gr.Blocks(title="Love Puzzle") as demo:
    gr.Markdown("# Love Puzzle\nEnter a phone number and three trivia domains to generate a puzzle.")
    with gr.Row():
        phone_input = gr.Textbox(label="Phone number", placeholder="e.g. (555) 867-5309")
    with gr.Row():
        d1 = gr.Textbox(label="Domain 1", placeholder="e.g. sports")
        d2 = gr.Textbox(label="Domain 2", placeholder="e.g. history")
        d3 = gr.Textbox(label="Domain 3", placeholder="e.g. music")
    btn = gr.Button("Generate Puzzle", variant="primary")
    output = gr.Textbox(label="Your Puzzle", lines=15)

    btn.click(fn=run, inputs=[phone_input, d1, d2, d3], outputs=output)

demo.launch()
