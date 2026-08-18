import json
import random
import spaces
import gradio as gr

from clue_generator import load_model, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from puzzle import generate_puzzle, validate_phone_number, format_equation

# Color pools each page load randomly draws from. Boxes/panels and buttons
# pick independently; input fields and borders are derived by lightening/
# darkening whichever box color got picked, not their own pool.
_BG_POOL = [
    "#EBB394", "#E8AE64", "#D1DB88", "#C1E09F", "#ACD9A3", "#A3D9AC",
    "#9CDBC9", "#9BCCE0", "#B5C2E8", "#C5B5E8", "#DCB5E8", "#E8B5E0", "#E6AEC7",
]
_BOX_POOL = ["#6E0E25", "#6E0E66", "#430E6E", "#0E1B6E", "#0E596E", "#0E6E41"]
_BUTTON_POOL = [
    "#BD0030", "#BD00B0", "#6E00BD", "#1800B3", "#005FB3",
    "#048A82", "#048A3C", "#1D8A04", "#8A3304",
]
_TEXT_COLOR = "#FFFFFF"
_INPUT_LIGHTEN = 0.18
_BORDER_DARKEN = 0.20
_HOVER_DARKEN = 0.12


def _blend(hex_color, target_rgb, amount):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    tr, tg, tb = target_rgb
    nr = round(r + (tr - r) * amount)
    ng = round(g + (tg - g) * amount)
    nb = round(b + (tb - b) * amount)
    return f"#{nr:02x}{ng:02x}{nb:02x}"


def _lighten(hex_color, amount=_INPUT_LIGHTEN):
    return _blend(hex_color, (255, 255, 255), amount)


def _darken(hex_color, amount=_BORDER_DARKEN):
    return _blend(hex_color, (0, 0, 0), amount)


_bg = random.choice(_BG_POOL)
_box = random.choice(_BOX_POOL)
_button_primary, _button_secondary = random.sample(_BUTTON_POOL, 2)
_theme = gr.themes.Soft().set(
    body_background_fill=_bg,
    body_text_color=_TEXT_COLOR,
    block_background_fill=_box,
    block_border_color=_darken(_box),
    block_border_width="2px",
    panel_background_fill=_box,
    panel_border_color=_darken(_box),
    block_label_background_fill=_button_primary,
    block_label_border_color=_darken(_button_primary),
    block_label_text_color=_TEXT_COLOR,
    block_title_text_color=_TEXT_COLOR,
    input_background_fill=_lighten(_box),
    input_border_color=_darken(_box),
    input_border_width="2px",
    button_primary_background_fill=_button_primary,
    button_primary_background_fill_hover=_darken(_button_primary, _HOVER_DARKEN),
    button_primary_border_color=_darken(_button_primary),
    button_primary_text_color=_TEXT_COLOR,
    button_secondary_background_fill=_button_secondary,
    button_secondary_background_fill_hover=_darken(_button_secondary, _HOVER_DARKEN),
    button_secondary_border_color=_darken(_button_secondary),
    button_secondary_text_color=_TEXT_COLOR,
)


# Gradio's theme= only picks colors once, when this process starts (baked into
# theme.css served to every visitor). To get a fresh random palette on every
# individual page load, we re-roll client-side.
#
# The deployed Gradio version (5.0.0, pinned by the HF Space's sdk_version)
# scopes its dark-mode variables to a bare `.dark { ... }` rule, and that
# class lands on a container element *below* <html>, not on <html> itself.
# That rules out overriding via inline style on document.documentElement:
# inline !important only outranks a stylesheet rule when both target the
# *same* element. Here they don't -- our value on <html> only reaches the
# .dark container by inheritance, and a direct (even non-important) rule on
# an element always wins over an inherited one, regardless of importance.
#
# So the override has to be a real stylesheet rule using the same selectors
# Gradio uses (:root and .dark), which matches wherever that class actually
# ends up. Selector + !important + later in the document beats theme.css's
# same-specificity, non-important rule. SvelteKit's hydration can reorder or
# re-insert <head> content after our script runs, so a MutationObserver
# re-appends our <style> tag to the end of <head> whenever that happens,
# keeping us last (and therefore winning) without re-rolling the colors.
_COLOR_HEAD = f"""
<script>
(function() {{
    var BG_POOL = {json.dumps(_BG_POOL)};
    var BOX_POOL = {json.dumps(_BOX_POOL)};
    var BUTTON_POOL = {json.dumps(_BUTTON_POOL)};
    var TEXT_COLOR = {json.dumps(_TEXT_COLOR)};
    var INPUT_LIGHTEN = {json.dumps(_INPUT_LIGHTEN)};
    var BORDER_DARKEN = {json.dumps(_BORDER_DARKEN)};
    var HOVER_DARKEN = {json.dumps(_HOVER_DARKEN)};

    function pick(arr) {{ return arr[Math.floor(Math.random() * arr.length)]; }}
    function pickTwo(arr) {{
        var pool = arr.slice();
        var a = pool.splice(Math.floor(Math.random() * pool.length), 1)[0];
        var b = pool.splice(Math.floor(Math.random() * pool.length), 1)[0];
        return [a, b];
    }}
    function blend(hex, target, amount) {{
        var r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
        var nr = Math.round(r + (target[0] - r) * amount);
        var ng = Math.round(g + (target[1] - g) * amount);
        var nb = Math.round(b + (target[2] - b) * amount);
        return '#' + [nr, ng, nb].map(function(v) {{ return v.toString(16).padStart(2, '0'); }}).join('');
    }}
    function lighten(hex, amount) {{ return blend(hex, [255, 255, 255], amount === undefined ? INPUT_LIGHTEN : amount); }}
    function darken(hex, amount) {{ return blend(hex, [0, 0, 0], amount === undefined ? BORDER_DARKEN : amount); }}

    var bg = pick(BG_POOL);
    var box = pick(BOX_POOL);
    var buttons = pickTwo(BUTTON_POOL);
    var buttonPrimary = buttons[0], buttonSecondary = buttons[1];

    var vars = {{
        '--body-background-fill': bg,
        '--body-text-color': TEXT_COLOR,
        '--block-background-fill': box,
        '--block-border-color': darken(box),
        '--block-border-width': '2px',
        '--panel-background-fill': box,
        '--panel-border-color': darken(box),
        '--block-label-background-fill': buttonPrimary,
        '--block-label-border-color': darken(buttonPrimary),
        '--block-label-text-color': TEXT_COLOR,
        '--block-title-text-color': TEXT_COLOR,
        '--input-background-fill': lighten(box),
        '--input-border-color': darken(box),
        '--input-border-width': '2px',
        '--button-primary-background-fill': buttonPrimary,
        '--button-primary-background-fill-hover': darken(buttonPrimary, HOVER_DARKEN),
        '--button-primary-border-color': darken(buttonPrimary),
        '--button-primary-text-color': TEXT_COLOR,
        '--button-secondary-background-fill': buttonSecondary,
        '--button-secondary-background-fill-hover': darken(buttonSecondary, HOVER_DARKEN),
        '--button-secondary-border-color': darken(buttonSecondary),
        '--button-secondary-text-color': TEXT_COLOR,
    }};
    var decls = Object.keys(vars).map(function(k) {{ return k + ': ' + vars[k] + ' !important;'; }}).join('\\n');
    var css = ':root, :root.dark, :root .dark, .dark {{\\n' + decls + '\\n}}';

    var style = null;
    function apply() {{
        if (!style || !style.isConnected) {{
            style = document.createElement('style');
            document.head.appendChild(style);
        }} else if (document.head.lastElementChild !== style) {{
            document.head.appendChild(style);
        }}
        style.textContent = css;
    }}

    apply();
    new MutationObserver(apply).observe(document.head, {{childList: true}});
}})();
</script>
"""

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
            return gr.update(), gr.update(), gr.update(value="Please enter a phone number.", visible=True), "", "", 0, False
        domains = [d for d in [domain1, domain2, domain3] if d]
        if len(domains) < 3:
            return gr.update(), gr.update(), gr.update(value="Please enter all three interests.", visible=True), "", "", 0, False

        try:
            phone = validate_phone_number(phone_raw)
        except ValueError as e:
            return gr.update(), gr.update(), gr.update(value=f"Invalid phone number: {e}", visible=True), "", "", 0, False

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
            gr.update(visible=False),               # hide page 1
            gr.update(visible=True),                # show page 2
            gr.update(value="", visible=False),      # clear+hide error
            puzzle,
            format_equation(eq),
            0,
            verify,
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        return gr.update(), gr.update(), gr.update(value=error_msg, visible=True), "", "", 0, False


def check_answer(guess: str, phone_raw: str, attempts: int, equation: str):
    try:
        phone = validate_phone_number(phone_raw.strip())
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
        gr.update(visible=True),               # show page 1
        gr.update(visible=False),              # hide page 2
        gr.update(value="", visible=False),    # clear+hide error
        "", "", 0, False,                      # reset state
    )


with gr.Blocks(title="Love Puzzle", theme=_theme, head=_COLOR_HEAD) as demo:
    equation_state = gr.State("")
    attempts_state = gr.State(0)
    verify_state = gr.State(False)

    # ── Page 1 ──────────────────────────────────────────────────────────────
    with gr.Column(visible=True) as page1:
        gr.Markdown("# 💌 Love Puzzle\nGenerate a puzzle from a phone number.")
        phone_input = gr.Textbox(label="Phone number", placeholder="e.g. (555) 867-5309")
        with gr.Row():
            d1 = gr.Textbox(label="Interest 1", placeholder="e.g. sports")
            d2 = gr.Textbox(label="Interest 2", placeholder="e.g. history")
            d3 = gr.Textbox(label="Interest 3", placeholder="e.g. music")
        verify_checkbox = gr.Checkbox(label="Enable answer verification on next page")
        error_output = gr.Textbox(label="", interactive=False, visible=False, show_label=False)
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
