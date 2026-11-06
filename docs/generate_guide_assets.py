import base64
from html import escape
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
assets = root / "docs" / "assets"
assets.mkdir(parents=True, exist_ok=True)
notebook = json.loads((root / "preprocessing.ipynb").read_text())
cell = next(
    cell for cell in notebook["cells"]
    if "display_unprocessed_spectrogram(train_dataset, 15)" in "".join(cell["source"])
)
images = [
    output["data"]["image/png"]
    for output in cell["outputs"]
    if "image/png" in output.get("data", {})
]
if len(images) != 2:
    raise ValueError("Expected the two saved notebook spectrogram figures.")
for name, image in zip(("air-conditioner", "gun-shot"), images):
    (assets / f"notebook-{name}.png").write_bytes(base64.b64decode(image))

panels = [
    ("Frequency masking", "frequency", ("Replace a band of rows.", "Keep the class label.")),
    ("Time masking", "time", ("Replace a run of columns.", "Keep the class label.")),
    ("MixUp", "mixup", ("Blend two images.", "Blend their labels with the same weights.")),
    ("CutMix", "cutmix", ("Replace a rectangle with another image.", "Mix labels according to the replaced area.")),
]
svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="700" viewBox="0 0 960 700" role="img" aria-labelledby="title description">',
    '<title id="title">Four ways to change a training spectrogram</title>',
    '<desc id="description">Frequency masking replaces rows. Time masking replaces columns. MixUp blends complete images and labels. CutMix replaces a rectangle and mixes labels by area. The grids are schematic, not dataset samples.</desc>',
    '<rect width="960" height="700" fill="#ffffff"/>',
    '<g font-family="Arial, Liberation Sans, sans-serif" fill="#172b3a">',
    '<text x="28" y="40" font-size="27" font-weight="700">What changes in the training image?</text>',
    '<text x="28" y="68" font-size="17">Schematic grids. Colored regions mark edits, not measured audio values.</text>',
]
for index, (title, operation, captions) in enumerate(panels):
    x = 28 + (index % 2) * 470
    y = 100 + (index // 2) * 292
    svg.append(f'<g transform="translate({x},{y})">')
    svg.append('<rect width="434" height="274" rx="8" fill="#f6f8fa" stroke="#d8e1e8"/>')
    svg.append(f'<text x="20" y="33" font-size="24" font-weight="700">{escape(title)}</text>')
    svg.append('<g transform="translate(55,56)">')
    for row in range(8):
        for column in range(12):
            value = (row * 7 + column * 3 + (row * column) % 11) % 15
            base = (40 + value * 5, 82 + value * 6, 116 + value * 6)
            if operation == "mixup":
                donor = (224, 146 + value * 3, 65 + value * 3)
                base = tuple(round(a * 0.6 + b * 0.4) for a, b in zip(base, donor))
            if operation == "cutmix" and 4 <= column < 10 and 1 <= row < 5:
                base = (224, 146 + value * 3, 65 + value * 3)
            color = "#" + "".join(f"{component:02x}" for component in base)
            svg.append(f'<rect x="{column * 28}" y="{row * 16}" width="28" height="16" fill="{color}"/>')
    if operation == "frequency":
        svg.append('<rect x="0" y="48" width="336" height="32" fill="#f4cc8b" stroke="#6b4715" stroke-width="2" stroke-dasharray="5 3"/>')
    elif operation == "time":
        svg.append('<rect x="140" y="0" width="56" height="128" fill="#f4cc8b" stroke="#6b4715" stroke-width="2" stroke-dasharray="5 3"/>')
    elif operation == "cutmix":
        svg.append('<rect x="112" y="16" width="168" height="64" fill="none" stroke="#6b4715" stroke-width="2" stroke-dasharray="5 3"/>')
    svg.extend([
        '<rect width="336" height="128" fill="none" stroke="#647787"/>',
        '<text x="168" y="150" text-anchor="middle" font-size="16">Time</text>',
        '<text transform="translate(-20,64) rotate(-90)" text-anchor="middle" font-size="16">Frequency</text>',
        '</g>',
    ])
    for line, caption in enumerate(captions):
        svg.append(f'<text x="20" y="{234 + line * 22}" font-size="17">{escape(caption)}</text>')
    svg.append('</g>')
svg.extend([
    '<text x="28" y="686" font-size="16" fill="#43586b">Masks run before resizing. MixUp or CutMix runs after normalization. Training only.</text>',
    '</g></svg>',
])
(assets / "spectrogram-augmentations.svg").write_text("\n".join(svg) + "\n")
print("Wrote two unchanged notebook PNGs and the augmentation schematic.")
