import spaces
import gradio as gr

from clue_generator import load_model, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from puzzle import generate_puzzle, _validate_phone

g_model, g_tokenizer = None, None
v_model, v_tokenizer = None, None


@spaces.GPU
def run(phone_raw: str, domain1: str, domain2: str, domain3: str):
    import traceback
    global g_model, g_tokenizer, v_model, v_tokenizer
    try:
        phone_raw = phone_raw.strip()
        domain1, domain2, domain3 = domain1.strip(), domain2.strip(), domain3.strip()

        if not phone_raw:
            return "Please enter a phone number.", "", 0
        domains = [d for d in [domain1, domain2, domain3] if d]
        if len(domains) < 3:
            return "Please enter all three domains.", "", 0

        try:
            phone = _validate_phone(phone_raw)
        except ValueError as e:
            return f"Invalid phone number: {e}", "", 0

        if g_model is None:
            print(f"Loading generator ({MODEL_ID})...")
            g_model, g_tokenizer = load_model()
        if v_model is None:
            print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
            v_model, v_tokenizer = load_verifier()

        eq, puzzle, _, _ = generate_puzzle(
            phone, domains, g_model, g_tokenizer, v_model, v_tokenizer
        )
        return puzzle, eq["infix"], 0
    except Exception as e:
        return f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}", "", 0


def check_answer(guess: str, phone_raw: str, attempts: int, equation: str):
    try:
        phone = _validate_phone(phone_raw.strip())
        guess_digits = guess.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if str(phone) == guess_digits:
            return "Correct! 🎉", attempts, gr.update(visible=False)
        else:
            new_attempts = attempts + 1
            msg = f"Wrong! ({new_attempts}/3 attempts used)"
            reveal_visible = new_attempts >= 3
            return msg, new_attempts, gr.update(visible=reveal_visible)
    except Exception as e:
        return f"Error: {e}", attempts, gr.update(visible=False)


def reveal(equation: str):
    return f"Equation: {equation}"


with gr.Blocks(title="Love Puzzle") as demo:
    equation_state = gr.State("")
    attempts_state = gr.State(0)

    gr.Markdown("# Love Puzzle\nEnter a phone number and three trivia domains to generate a puzzle.")
    with gr.Row():
        phone_input = gr.Textbox(label="Phone number", placeholder="e.g. (555) 867-5309")
    with gr.Row():
        d1 = gr.Textbox(label="Domain 1", placeholder="e.g. sports")
        d2 = gr.Textbox(label="Domain 2", placeholder="e.g. history")
        d3 = gr.Textbox(label="Domain 3", placeholder="e.g. music")
    generate_btn = gr.Button("Generate Puzzle", variant="primary")
    puzzle_output = gr.Textbox(label="Your Puzzle", lines=15)

    gr.Markdown("### Solve it")
    guess_input = gr.Textbox(label="What's the phone number?", placeholder="e.g. (555) 867-5309")
    check_btn = gr.Button("Check Answer")
    result_output = gr.Textbox(label="Result", interactive=False)
    reveal_btn = gr.Button("Reveal Equation", variant="secondary", visible=False)
    equation_output = gr.Textbox(label="Equation", interactive=False, visible=True)

    generate_btn.click(
        fn=run,
        inputs=[phone_input, d1, d2, d3],
        outputs=[puzzle_output, equation_state, attempts_state],
    )

    check_btn.click(
        fn=check_answer,
        inputs=[guess_input, phone_input, attempts_state, equation_state],
        outputs=[result_output, attempts_state, reveal_btn],
    )

    reveal_btn.click(
        fn=reveal,
        inputs=[equation_state],
        outputs=[equation_output],
    )

demo.launch()
