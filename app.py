import spaces
import streamlit as st

from clue_generator import load_model, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from puzzle import generate_puzzle, _validate_phone

st.set_page_config(page_title="Love Puzzle", page_icon="💌")

_g_model, _g_tokenizer = None, None
_v_model, _v_tokenizer = None, None


@spaces.GPU
def run(phone_raw: str, domain1: str, domain2: str, domain3: str):
    global _g_model, _g_tokenizer, _v_model, _v_tokenizer
    phone = _validate_phone(phone_raw.strip())
    domains = [d.strip() for d in [domain1, domain2, domain3]]
    if _g_model is None:
        _g_model, _g_tokenizer = load_model()
    if _v_model is None:
        _v_model, _v_tokenizer = load_verifier()
    eq, puzzle, _, _ = generate_puzzle(
        phone, domains, _g_model, _g_tokenizer, _v_model, _v_tokenizer
    )
    return puzzle, eq["infix"]


for _k, _v in {
    "page": 1, "puzzle": "", "equation": "", "attempts": 0,
    "do_verify": False, "phone_raw": "", "result_msg": "",
    "show_reveal": False, "equation_revealed": False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Page 1 ───────────────────────────────────────────────────────────────────
if st.session_state.page == 1:
    st.title("💌 Love Puzzle")
    st.caption("Generate a math puzzle from a phone number.")

    phone_raw = st.text_input("Phone number", placeholder="e.g. (555) 867-5309")
    c1, c2, c3 = st.columns(3)
    with c1:
        d1 = st.text_input("Domain 1", placeholder="e.g. sports")
    with c2:
        d2 = st.text_input("Domain 2", placeholder="e.g. history")
    with c3:
        d3 = st.text_input("Domain 3", placeholder="e.g. music")
    do_verify = st.checkbox("Enable answer verification")

    if st.button("Generate Puzzle ➜", type="primary"):
        if not phone_raw.strip():
            st.error("Please enter a phone number.")
        elif not all([d1.strip(), d2.strip(), d3.strip()]):
            st.error("Please enter all three domains.")
        else:
            try:
                with st.spinner("Generating your puzzle…"):
                    puzzle, equation = run(phone_raw, d1, d2, d3)
                st.session_state.update({
                    "page": 2, "puzzle": puzzle, "equation": equation,
                    "do_verify": do_verify, "phone_raw": phone_raw,
                    "attempts": 0, "result_msg": "",
                    "show_reveal": False, "equation_revealed": False,
                })
                st.rerun()
            except ValueError as e:
                st.error(f"Invalid phone number: {e}")
            except Exception as e:
                import traceback
                st.error(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


# ── Page 2 ───────────────────────────────────────────────────────────────────
elif st.session_state.page == 2:
    st.title("💌 Your Puzzle")
    st.text(st.session_state.puzzle)
    st.divider()

    if st.session_state.do_verify:
        st.subheader("Think you know the number?")
        guess = st.text_input("Enter the phone number", placeholder="e.g. (555) 867-5309")

        if st.button("Check Answer", type="primary"):
            try:
                phone = _validate_phone(st.session_state.phone_raw.strip())
                guess_clean = (guess.strip()
                               .replace(" ", "").replace("-", "")
                               .replace("(", "").replace(")", ""))
                if str(phone) == guess_clean:
                    st.session_state.result_msg = "✅ Correct!"
                else:
                    st.session_state.attempts += 1
                    st.session_state.result_msg = (
                        f"❌ Wrong! ({st.session_state.attempts}/3 attempts used)"
                    )
                    if st.session_state.attempts >= 3:
                        st.session_state.show_reveal = True
            except Exception as e:
                st.session_state.result_msg = f"Error: {e}"
            st.rerun()

        if st.session_state.result_msg:
            if st.session_state.result_msg.startswith("✅"):
                st.success(st.session_state.result_msg)
            else:
                st.warning(st.session_state.result_msg)

        if st.session_state.show_reveal:
            if st.button("Reveal Equation"):
                st.session_state.equation_revealed = True
                st.rerun()

        if st.session_state.equation_revealed:
            st.info(f"Equation: {st.session_state.equation}")

        st.divider()

    if st.button("← Start Over"):
        st.session_state.page = 1
        st.rerun()
