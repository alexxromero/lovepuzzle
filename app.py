import json
import random
import spaces
import gradio as gr

from clue_generator import load_model, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from puzzle import (
    generate_puzzle, validate_phone_number, format_equation, fact_check_summary,
    encode_puzzle, decode_puzzle,
)
from fact_checker import SERPER_API_KEY

# Everything derives from a single randomly-picked base color per page load,
# so the palette is always in harmony instead of clashing: box = base
# lightened once, input = base lightened again on top of that, background =
# base's complementary hue, lightened heavily into a pale tint.
_BASE_POOL = [
    "#A60808", "#056962", "#0A5C7D", "#130B8C", "#2A0582", "#4A0582",
    "#580582", "#820578", "#820550", "#820537", "#82051E",
]
_TEXT_COLOR = "#FFFFFF"
_BOX_LIGHTEN = 0.35
_INPUT_LIGHTEN = 0.35
_BG_LIGHTEN = 0.75
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


def _lighten(hex_color, amount):
    return _blend(hex_color, (255, 255, 255), amount)


def _darken(hex_color, amount=_BORDER_DARKEN):
    return _blend(hex_color, (0, 0, 0), amount)


def _rgb_to_hsl(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h / 6, s, l


def _hsl_to_rgb(h, s, l):
    if s == 0:
        r = g = b = l
    else:
        def hue2rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1 / 3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1 / 3)
    return round(r * 255), round(g * 255), round(b * 255)


def _complementary(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    h, s, l = _rgb_to_hsl(r, g, b)
    r2, g2, b2 = _hsl_to_rgb((h + 0.5) % 1.0, s, l)
    return f"#{r2:02x}{g2:02x}{b2:02x}"


_base = random.choice(_BASE_POOL)
_box = _lighten(_base, _BOX_LIGHTEN)
_input_color = _lighten(_box, _INPUT_LIGHTEN)
_bg = _lighten(_complementary(_base), _BG_LIGHTEN)
_theme = gr.themes.Soft().set(
    body_background_fill=_bg,
    body_text_color=_TEXT_COLOR,
    block_background_fill=_box,
    block_border_color=_darken(_box),
    block_border_width="2px",
    panel_background_fill=_box,
    panel_border_color=_darken(_box),
    block_label_background_fill=_base,
    block_label_border_color=_darken(_base),
    block_label_text_color=_TEXT_COLOR,
    block_title_text_color=_TEXT_COLOR,
    input_background_fill=_input_color,
    input_border_color=_darken(_box),
    input_border_width="2px",
    button_primary_background_fill=_base,
    button_primary_background_fill_hover=_darken(_base, _HOVER_DARKEN),
    button_primary_border_color=_darken(_base),
    button_primary_text_color=_TEXT_COLOR,
    button_secondary_background_fill=_base,
    button_secondary_background_fill_hover=_darken(_base, _HOVER_DARKEN),
    button_secondary_border_color=_darken(_base),
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
    var BASE_POOL = {json.dumps(_BASE_POOL)};
    var TEXT_COLOR = {json.dumps(_TEXT_COLOR)};
    var BOX_LIGHTEN = {json.dumps(_BOX_LIGHTEN)};
    var INPUT_LIGHTEN = {json.dumps(_INPUT_LIGHTEN)};
    var BG_LIGHTEN = {json.dumps(_BG_LIGHTEN)};
    var BORDER_DARKEN = {json.dumps(_BORDER_DARKEN)};
    var HOVER_DARKEN = {json.dumps(_HOVER_DARKEN)};

    function pick(arr) {{ return arr[Math.floor(Math.random() * arr.length)]; }}
    function blend(hex, target, amount) {{
        var r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
        var nr = Math.round(r + (target[0] - r) * amount);
        var ng = Math.round(g + (target[1] - g) * amount);
        var nb = Math.round(b + (target[2] - b) * amount);
        return '#' + [nr, ng, nb].map(function(v) {{ return v.toString(16).padStart(2, '0'); }}).join('');
    }}
    function lighten(hex, amount) {{ return blend(hex, [255, 255, 255], amount); }}
    function darken(hex, amount) {{ return blend(hex, [0, 0, 0], amount === undefined ? BORDER_DARKEN : amount); }}

    function rgbToHsl(r, g, b) {{
        r /= 255; g /= 255; b /= 255;
        var mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        var l = (mx + mn) / 2;
        if (mx === mn) return [0, 0, l];
        var d = mx - mn;
        var s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
        var h;
        if (mx === r) h = (g - b) / d + (g < b ? 6 : 0);
        else if (mx === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        return [h / 6, s, l];
    }}
    function hue2rgb(p, q, t) {{
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    }}
    function hslToRgb(h, s, l) {{
        var r, g, b;
        if (s === 0) {{
            r = g = b = l;
        }} else {{
            var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            var p = 2 * l - q;
            r = hue2rgb(p, q, h + 1 / 3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1 / 3);
        }}
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
    }}
    function complementary(hex) {{
        var r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
        var hsl = rgbToHsl(r, g, b);
        var rgb = hslToRgb((hsl[0] + 0.5) % 1.0, hsl[1], hsl[2]);
        return '#' + rgb.map(function(v) {{ return v.toString(16).padStart(2, '0'); }}).join('');
    }}

    var base = pick(BASE_POOL);
    var box = lighten(base, BOX_LIGHTEN);
    var inputColor = lighten(box, INPUT_LIGHTEN);
    var bg = lighten(complementary(base), BG_LIGHTEN);

    var vars = {{
        '--body-background-fill': bg,
        '--body-text-color': TEXT_COLOR,
        '--block-background-fill': box,
        '--block-border-color': darken(box),
        '--block-border-width': '2px',
        '--panel-background-fill': box,
        '--panel-border-color': darken(box),
        '--block-label-background-fill': base,
        '--block-label-border-color': darken(base),
        '--block-label-text-color': TEXT_COLOR,
        '--block-title-text-color': TEXT_COLOR,
        '--input-background-fill': inputColor,
        '--input-border-color': darken(box),
        '--input-border-width': '2px',
        '--button-primary-background-fill': base,
        '--button-primary-background-fill-hover': darken(base, HOVER_DARKEN),
        '--button-primary-border-color': darken(base),
        '--button-primary-text-color': TEXT_COLOR,
        '--button-secondary-background-fill': base,
        '--button-secondary-background-fill-hover': darken(base, HOVER_DARKEN),
        '--button-secondary-border-color': darken(base),
        '--button-secondary-text-color': TEXT_COLOR,
    }};
    var decls = Object.keys(vars).map(function(k) {{ return k + ': ' + vars[k] + ' !important;'; }}).join('\\n');
    // The deployed Gradio version (5.0.0) never wires up --body-background-fill
    // to any real element -- that only landed in a later Gradio release, which
    // makes the variable inert here. Paint body/html directly instead of
    // relying on Gradio's own (missing, in this version) consuming CSS rule.
    // Typed input text has no theme variable at all (only size/weight), so it
    // falls back to some default that isn't guaranteed readable against our
    // colored input fill -- force it directly too. ::placeholder isn't
    // targeted by `color` on the input itself, so placeholders keep their own
    // (muted) theme color.
    var css = ':root, :root.dark, :root .dark, .dark {{\\n' + decls + '\\n}}'
        + '\\nhtml, body {{ background: ' + bg + ' !important; color: ' + TEXT_COLOR + ' !important; }}'
        + '\\ninput, textarea {{ color: ' + base + ' !important; }}'
        + '\\ninput::placeholder, textarea::placeholder {{ color: ' + TEXT_COLOR + ' !important; opacity: 1 !important; }}'
        + '\\n#page1-title, #page1-title * {{ color: ' + base + ' !important; }}';

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


def _fact_check_status(clues_info):
    """Human-readable line showing whether/how much the search API contributed
    to this puzzle's generated clues."""
    if not SERPER_API_KEY:
        return "🔍 Search fact-check: off (no API key configured)"
    n_confirmed, n_generated = fact_check_summary(clues_info)
    if n_generated == 0:
        return "🔍 Search fact-check: on — no generated clues needed checking this time"
    return f"🔍 Search fact-check: on — {n_confirmed}/{n_generated} generated clues confirmed via search"


def _request_origin(request: gr.Request):
    """Best-effort scheme+host for building a shareable absolute URL. Reads
    the Host/X-Forwarded-Proto headers directly rather than trusting
    starlette's request.base_url, since that can reflect the container's
    internal address instead of the public one behind HF Spaces' proxy.
    """
    if request is None:
        return ""
    headers = dict(request.headers)
    host = headers.get("host", "")
    if not host:
        return ""
    proto = headers.get("x-forwarded-proto", "https")
    return f"{proto}://{host}"


@spaces.GPU
def run(phone_raw: str, domain1: str, domain2: str, domain3: str, verify: bool):
    import traceback
    global g_model, g_tokenizer, v_model, v_tokenizer
    try:
        phone_raw = phone_raw.strip()
        domain1, domain2, domain3 = domain1.strip(), domain2.strip(), domain3.strip()

        if not phone_raw:
            return gr.update(), gr.update(), gr.update(value="Please enter a phone number.", visible=True), "", "", 0, False, ""
        domains = [d for d in [domain1, domain2, domain3] if d]
        if len(domains) < 3:
            return gr.update(), gr.update(), gr.update(value="Please enter all three interests.", visible=True), "", "", 0, False, ""

        try:
            phone = validate_phone_number(phone_raw)
        except ValueError as e:
            return gr.update(), gr.update(), gr.update(value=f"Invalid phone number: {e}", visible=True), "", "", 0, False, ""

        if g_model is None:
            print(f"Loading generator ({MODEL_ID})...")
            g_model, g_tokenizer = load_model()
        if v_model is None:
            print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
            v_model, v_tokenizer = load_verifier()

        eq, puzzle, clues_info, _ = generate_puzzle(
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
            _fact_check_status(clues_info),
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        return gr.update(), gr.update(), gr.update(value=error_msg, visible=True), "", "", 0, False, ""


def check_answer(guess: str, phone_raw: str, attempts: int, equation: str):
    try:
        phone = validate_phone_number(phone_raw.strip())
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
    return gr.update(value=f"Equation: {equation}", visible=True)


def start_over():
    return (
        gr.update(visible=True),               # show page 1
        gr.update(visible=False),              # hide page 2
        gr.update(value="", visible=False),    # clear+hide error
        "", "", 0, False,                      # reset state
        "",                                    # clear fact-check status
        gr.update(value="", visible=False),    # clear+hide share link
    )


def on_page_load(request: gr.Request):
    """Jump straight to page 2, fully populated, when the URL carries a
    shared puzzle token (?p=...) -- lets a second person open the link and
    solve it without ever seeing page 1 or re-running any models."""
    token = dict(request.query_params).get("p") if request else None
    if token:
        try:
            puzzle_text, phone, equation_str = decode_puzzle(token)
            return (
                gr.update(visible=False),      # hide page 1
                gr.update(visible=True),       # show page 2
                puzzle_text,                   # puzzle_output
                equation_str,                  # equation_state
                str(phone),                    # phone_input (used by check_answer)
                0,                             # attempts_state
                True,                          # verify_state
                gr.update(visible=True),       # verify_section -- the whole point of the link
            )
        except ValueError:
            pass  # malformed token -- fall through to the normal page 1 view
    return gr.update(), gr.update(), "", "", "", 0, False, gr.update(visible=False)


with gr.Blocks(title="Love Puzzle", theme=_theme, head=_COLOR_HEAD) as demo:
    equation_state = gr.State("")
    attempts_state = gr.State(0)
    verify_state = gr.State(False)

    # ── Page 1 ──────────────────────────────────────────────────────────────
    with gr.Column(visible=True) as page1:
        gr.Markdown("# 💌 Love Puzzle\nGenerate a puzzle from a phone number.", elem_id="page1-title")
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
        share_url_output = gr.Textbox(label="Share this puzzle", interactive=False, visible=False)
        fact_check_output = gr.Markdown("", container=True)

        with gr.Column(visible=False) as verify_section:
            gr.Markdown("### Think you know the number?")
            guess_input = gr.Textbox(label="Enter the phone number", placeholder="e.g. (555) 867-5309")
            check_btn = gr.Button("Check Answer", variant="primary")
            result_output = gr.Textbox(label="", interactive=False, show_label=False)
            reveal_btn = gr.Button("Reveal Equation", variant="secondary", visible=False)
            equation_output = gr.Textbox(label="", interactive=False, show_label=False, visible=False)

        back_btn = gr.Button("← Start Over", variant="secondary")

    # ── Wiring ───────────────────────────────────────────────────────────────
    def on_generate(phone_raw, d1, d2, d3, verify, request: gr.Request):
        result = run(phone_raw, d1, d2, d3, verify)
        # result: (page1_update, page2_update, error, puzzle, equation, attempts, verify, fact_check_status)
        page1_upd, page2_upd, error, puzzle, equation, attempts, verify_val, fact_check_status = result
        verify_section_upd = gr.update(visible=verify_val)

        share_url_upd = gr.update(value="", visible=False)
        if puzzle:
            try:
                phone = validate_phone_number(phone_raw.strip())
                token = encode_puzzle(puzzle, phone, equation)
                origin = _request_origin(request)
                if origin:
                    share_url_upd = gr.update(value=f"{origin}/?p={token}", visible=True)
            except Exception:
                pass  # keep the puzzle usable even if the share link can't be built

        return (
            page1_upd, page2_upd, error, puzzle, equation, attempts,
            verify_val, verify_section_upd, fact_check_status, share_url_upd,
        )

    generate_btn.click(
        fn=on_generate,
        inputs=[phone_input, d1, d2, d3, verify_checkbox],
        outputs=[page1, page2, error_output, puzzle_output, equation_state, attempts_state, verify_state, verify_section, fact_check_output, share_url_output],
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

    back_btn.click(
        fn=start_over,
        inputs=[],
        outputs=[page1, page2, error_output, puzzle_output, equation_state, attempts_state, verify_state, fact_check_output, share_url_output],
    )

    demo.load(
        fn=on_page_load,
        inputs=None,
        outputs=[page1, page2, puzzle_output, equation_state, phone_input, attempts_state, verify_state, verify_section],
    )

demo.launch()
