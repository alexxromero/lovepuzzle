import random
import gradio as gr
from gradio.themes.utils import colors as c

VIBRANT = {
    "red": c.red, "pink": c.pink, "fuchsia": c.fuchsia, "purple": c.purple,
    "indigo": c.indigo, "blue": c.blue, "cyan": c.cyan, "teal": c.teal,
    "emerald": c.emerald, "green": c.green, "lime": c.lime,
    "yellow": c.yellow, "amber": c.amber, "orange": c.orange,
}
NEUTRALS = {
    "slate": c.slate, "gray": c.gray, "zinc": c.zinc,
    "neutral": c.neutral, "stone": c.stone,
}

primary_name, secondary_name = random.sample(sorted(VIBRANT), 2)
neutral_name = random.choice(sorted(NEUTRALS))

theme = gr.themes.Soft(
    primary_hue=VIBRANT[primary_name],
    secondary_hue=VIBRANT[secondary_name],
    neutral_hue=NEUTRALS[neutral_name],
).set(
    background_fill_primary="*primary_50",
    background_fill_secondary="*secondary_50",
)

print(f"\nTheme code to paste into app.py:")
print(f"  theme=gr.themes.Soft(")
print(f'      primary_hue="{primary_name}",')
print(f'      secondary_hue="{secondary_name}",')
print(f'      neutral_hue="{neutral_name}",')
print(f"  )\n")
print("Restart this script to generate new colors.\n")

SAMPLE_PUZZLE = """\
1. Start with 6.
2. Multiply by: the number of players on a basketball team, plus 2.
3. Add the number of strings on a guitar.
4. Subtract the number of days in a week.
5. Multiply by the number of continents on Earth."""

with gr.Blocks(title="Theme Preview — Love Puzzle") as demo:

    with gr.Column(visible=True) as page1:
        gr.Markdown("# 💌 Love Puzzle\nGenerate a puzzle from a phone number.")
        gr.Textbox(label="Phone number", placeholder="e.g. (555) 867-5309")
        with gr.Row():
            gr.Textbox(label="Domain 1", placeholder="e.g. sports")
            gr.Textbox(label="Domain 2", placeholder="e.g. history")
            gr.Textbox(label="Domain 3", placeholder="e.g. music")
        gr.Checkbox(label="Enable answer verification on next page")
        gr.Button("Generate Puzzle ➜", variant="primary")

    gr.Markdown("---")

    with gr.Column(visible=True) as page2:
        gr.Markdown("# 💌 Your Puzzle")
        gr.Textbox(value=SAMPLE_PUZZLE, lines=8, interactive=False, show_label=False)
        with gr.Column():
            gr.Markdown("### Think you know the number?")
            gr.Textbox(label="Enter the phone number", placeholder="e.g. (555) 867-5309")
            gr.Button("Check Answer", variant="primary")
            gr.Textbox(value="❌ Wrong! (1/3 attempts used)", interactive=False, show_label=False)
            gr.Button("Reveal Equation", variant="secondary")
        gr.Button("← Start Over", variant="secondary")

demo.launch(theme=theme)
