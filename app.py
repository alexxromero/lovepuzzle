import spaces
import gradio as gr

from clue_generator import load_model, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from puzzle import generate_puzzle, _validate_phone

g_model, g_tokenizer = None, None
v_model, v_tokenizer = None, None


@spaces.GPU
def run(phone_raw: str, domain1: str, domain2: str, domain3: str, verify: bool):
    import traceback
    global g_model, g_tokenizer, v_model, v_tokenizer
    try:
        phone_raw = phone_raw.strip()
        domain1, domain2, domain3 = domain1.strip(), domain2.strip(), domain3.strip()

        if not phone_raw:
            return gr.update(), gr.update(), "Please enter a phone number.", "", "", 0, False
        domains = [d for d in [domain1, domain2, domain3] if d]
        if len(domains) < 3:
            return gr.update(), gr.update(), "Please enter all three domains.", "", "", 0, False

        try:
            phone = _validate_phone(phone_raw)
        except ValueError as e:
            return gr.update(), gr.update(), f"Invalid phone number: {e}", "", "", 0, False

        if g_model is None:
            print(f"Loading generator ({MODEL_ID})...")
            g_model, g_tokenizer = load_model()
        if v_model is None:
            print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
            v_model, v_tokenizer = load_verifier()

        eq, puzzle, _, _ = generate_puzzle(
            phone, domains, g_model, g_tokenizer, v_model, v_tokenizer
        )
        return (
            gr.update(visible=False),   # hide page 1
            gr.update(visible=True),    # show page 2
            "",                         # clear error
            puzzle,
            eq["infix"],
            0,
            verify,
        )
    except Exception as e:
        import traceback
        return gr.update(), gr.update(), f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}", "", "", 0, False


def check_answer(guess: str, phone_raw: str, attempts: int, equation: str):
    try:
        phone = _validate_phone(phone_raw.strip())
        guess_digits = guess.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if str(phone) == guess_digits:
            return "Correct! 🎉", attempts, gr.update(visible=False), gr.update(visible=False)
        else:
            new_attempts = attempts + 1
            msg = f"Wrong! ({new_attempts}/3 attempts used)"
            reveal_visible = new_attempts >= 3
            return msg, new_attempts, gr.update(visible=reveal_visible), gr.update(visible=True)
    except Exception as e:
        return f"Error: {e}", attempts, gr.update(visible=False), gr.update(visible=True)


def reveal(equation: str):
    return gr.update(value=f"Equation: {equation}", visible=True)


def start_over():
    return (
        gr.update(visible=True),   # show page 1
        gr.update(visible=False),  # hide page 2
        "",                        # clear error
        "", "", 0, False,          # reset state
    )


with gr.Blocks(title="Love Puzzle") as demo:
    equation_state = gr.State("")
    attempts_state = gr.State(0)
    verify_state = gr.State(False)

    # ── Page 1 ──────────────────────────────────────────────────────────────
    with gr.Column(visible=True) as page1:
        gr.Markdown("# 💌 Love Puzzle\nGenerate a puzzle from a phone number.")
        phone_input = gr.Textbox(label="Phone number", placeholder="e.g. (555) 867-5309")
        with gr.Row():
            d1 = gr.Textbox(label="Domain 1", placeholder="e.g. sports")
            d2 = gr.Textbox(label="Domain 2", placeholder="e.g. history")
            d3 = gr.Textbox(label="Domain 3", placeholder="e.g. music")
        verify_checkbox = gr.Checkbox(label="Enable answer verification on next page")
        error_output = gr.Textbox(label="", interactive=False, visible=True, show_label=False)
        generate_btn = gr.Button("Generate Puzzle ➜", variant="primary")

    # ── Page 2 ──────────────────────────────────────────────────────────────
    with gr.Column(visible=False) as page2:
        gr.Markdown("# 💌 Your Puzzle")
        puzzle_output = gr.Textbox(label="", lines=15, interactive=False, show_label=False)

        with gr.Column(visible=False) as verify_section:
            gr.Markdown("### Think you know the number?")
            guess_input = gr.Textbox(label="Enter the phone number", placeholder="e.g. (555) 867-5309")
            check_btn = gr.Button("Check Answer", variant="primary")
            result_output = gr.Textbox(label="", interactive=False, show_label=False)
            reveal_btn = gr.Button("Reveal Equation", variant="secondary", visible=False)
            equation_output = gr.Textbox(label="", interactive=False, show_label=False, visible=False)

        back_btn = gr.Button("← Start Over", variant="secondary")

    # ── Wiring ───────────────────────────────────────────────────────────────
    def on_generate(phone_raw, d1, d2, d3, verify):
        result = run(phone_raw, d1, d2, d3, verify)
        # result: (page1_update, page2_update, error, puzzle, equation, attempts, verify)
        page1_upd, page2_upd, error, puzzle, equation, attempts, verify_val = result
        verify_section_upd = gr.update(visible=verify_val)
        return page1_upd, page2_upd, error, puzzle, equation, attempts, verify_val, verify_section_upd

    generate_btn.click(
        fn=on_generate,
        inputs=[phone_input, d1, d2, d3, verify_checkbox],
        outputs=[page1, page2, error_output, puzzle_output, equation_state, attempts_state, verify_state, verify_section],
    )

    check_btn.click(
        fn=check_answer,
        inputs=[guess_input, phone_input, attempts_state, equation_state],
        outputs=[result_output, attempts_state, reveal_btn, result_output],
    )

    reveal_btn.click(
        fn=reveal,
        inputs=[equation_state],
        outputs=[equation_output],
    )

    back_btn.click(
        fn=start_over,
        inputs=[],
        outputs=[page1, page2, error_output, puzzle_output, equation_state, attempts_state, verify_state],
    )

demo.launch()
