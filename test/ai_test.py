import json
import numpy as np
from PIL import Image
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

IMAGE_PATH = "image/sample/screenshot1.png"

army_1_file = "army_strategies/army_1.json"
with open(army_1_file, 'r') as f:
    army_1 = json.load(f)

print(army_1)

# Resource boundary coords from 4 clicks:
# Click1:(45,139) Click2:(360,135) Click3:(34,401) Click4:(319,400)
RESOURCE_BOUNDS = (34, 135, 360, 401)

# How many pixels to skip on the left to cut out the resource icon
ICON_WIDTH = 60

# Tesseract config: single line, digits only
OCR_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789'


def preprocess_row(row_img: Image.Image) -> Image.Image:
    """Isolate bright (white/cream) number pixels for cleaner OCR."""
    arr = np.array(row_img.convert("RGB"))
    brightness = arr.mean(axis=2)
    mask = brightness > 160
    result = np.zeros_like(arr)
    result[mask] = 255
    return Image.fromarray(result.astype(np.uint8)).convert("L")


def extract_resources(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")

    x1, y1, x2, y2 = RESOURCE_BOUNDS
    crop = img.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size

    # Top ~45px is the "Available Loot:" title; three equal rows below it
    row_top = 45
    row_h = (crop_h - row_top) // 3

    resources = {}
    for i, name in enumerate(["gold", "elixir", "dark_elixir"]):
        ry1 = row_top + i * row_h
        ry2 = row_top + (i + 1) * row_h
        row = crop.crop((ICON_WIDTH, ry1, crop_w, ry2))

        processed = preprocess_row(row)
        raw = pytesseract.image_to_string(processed, config=OCR_CONFIG).strip()

        resources[name] = int(raw) if raw.isdigit() else 0

    return resources


img = Image.open(IMAGE_PATH)
data = extract_resources(IMAGE_PATH)

test_txt = "test/test.txt"
with open(test_txt, "w") as f:
    f.write(json.dumps(data, indent=2))

print(f"Gold:        {data.get('gold', 0):,}")
print(f"Elixir:      {data.get('elixir', 0):,}")
print(f"Dark Elixir: {data.get('dark_elixir', 0):,}")
