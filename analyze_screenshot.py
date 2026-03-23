import os
import json
import base64
import io
import boto3
import numpy as np
import pytesseract
from PIL import Image
from pynput import keyboard as pynput_keyboard
from dotenv import load_dotenv
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Boundary from user-defined 4-click region around "Available Loot"
# Click1:(45,139) Click2:(360,135) Click3:(34,401) Click4:(319,400)
_RESOURCE_BOUNDS = (34, 135, 360, 401)
_ICON_WIDTH      = 60   # pixels to skip on left (resource icon)
_ROW_TOP         = 45   # pixels to skip at top ("Available Loot:" title)
_OCR_CONFIG      = "--psm 7 -c tessedit_char_whitelist=0123456789"


def _preprocess_row(row_img: Image.Image) -> Image.Image:
    """Isolate bright (white/cream) number pixels for cleaner OCR."""
    arr = np.array(row_img.convert("RGB"))
    mask = arr.mean(axis=2) > 160
    result = np.zeros_like(arr)
    result[mask] = 255
    return Image.fromarray(result.astype(np.uint8)).convert("L")


_USER_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789,"


def analyze_user_resources(screenshot: Image.Image, bounds: tuple) -> dict:
    """
    Extract the player's own Gold / Elixir / Dark Elixir from the
    top-right storage panel visible on the scout/home screen.
    bounds: (x1, y1, x2, y2) pixel coords of the panel.
    Returns: {"gold": int, "elixir": int, "dark_elixir": int}
    """
    x1, y1, x2, y2 = bounds
    crop = screenshot.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size
    row_h = crop_h // 3
    result = {}
    for i, name in enumerate(("gold", "elixir", "dark_elixir")):
        ry1 = i * row_h
        ry2 = (i + 1) * row_h
        row = crop.crop((_ICON_WIDTH, ry1, crop_w, ry2))
        raw = pytesseract.image_to_string(_preprocess_row(row), config=_USER_OCR_CONFIG).strip()
        raw = raw.replace(",", "")
        result[name] = int(raw) if raw.isdigit() else 0
    return result


def analyze_user_resources_with_ai(screenshot: Image.Image, bounds: tuple) -> dict:
    """Crop the user storage panel and send to Bedrock AI to read Gold/Elixir/Dark Elixir.
    Returns: {"gold": int, "elixir": int, "dark_elixir": int}
    """
    x1, y1, x2, y2 = bounds
    crop = screenshot.crop((x1, y1, x2, y2))
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = (
        "This is a cropped region from a Clash of Clans screenshot showing the player's "
        "own resource storage amounts (top-right panel). "
        "Extract the current Gold, Elixir, and Dark Elixir amounts shown. "
        "Return ONLY a JSON object with keys: gold, elixir, dark_elixir (integers, 0 if not visible). "
        'Example: {"gold": 15000000, "elixir": 12000000, "dark_elixir": 250000}'
    )
    body = json.dumps({
        "messages": [{"role": "user", "content": [
            {"image": {"format": "png", "source": {"bytes": img_b64}}},
            {"text": prompt},
        ]}],
        "inferenceConfig": {"max_new_tokens": 256, "temperature": 0.1},
    })
    response = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
    result_body = json.loads(response["body"].read())
    raw_text = result_body["output"]["message"]["content"][0]["text"].strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"gold": 0, "elixir": 0, "dark_elixir": 0}


def validate_storage_full_with_ai(screenshot: Image.Image, bounds: tuple) -> dict:
    """Crop the user storage panel and ask Bedrock AI which resources are at max capacity.
    Returns: {"gold_full": bool, "elixir_full": bool, "dark_elixir_full": bool}
    """
    x1, y1, x2, y2 = bounds
    crop = screenshot.crop((x1, y1, x2, y2))
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = (
        "This is a cropped region from a Clash of Clans home village screenshot "
        "showing the player's resource storage panel. "
        "Determine which resources appear to be at maximum (full) capacity. "
        "Return ONLY a JSON object with keys: gold_full, elixir_full, dark_elixir_full (booleans). "
        'Example: {"gold_full": true, "elixir_full": true, "dark_elixir_full": false}'
    )
    body = json.dumps({
        "messages": [{"role": "user", "content": [
            {"image": {"format": "png", "source": {"bytes": img_b64}}},
            {"text": prompt},
        ]}],
        "inferenceConfig": {"max_new_tokens": 256, "temperature": 0.1},
    })
    response = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
    result_body = json.loads(response["body"].read())
    raw_text = result_body["output"]["message"]["content"][0]["text"].strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"gold_full": False, "elixir_full": False, "dark_elixir_full": False}


def read_looted_resources_ocr(screenshot: Image.Image, bounds: tuple) -> dict:
    """OCR the post-battle loot popup to read how much Gold/Elixir/Dark Elixir was looted.
    Returns: {"gold": int, "elixir": int, "dark_elixir": int}
    """
    x1, y1, x2, y2 = bounds
    crop = screenshot.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size
    row_h = crop_h // 3
    # Loot popup has no icons — do not strip from left.
    # psm 8 (single word) handles the spaced thousands separator better than psm 7.
    _LOOT_OCR_CONFIG = "--psm 8 -c tessedit_char_whitelist=0123456789"
    result = {}
    for i, name in enumerate(("gold", "elixir", "dark_elixir")):
        ry1 = i * row_h
        ry2 = (i + 1) * row_h
        row = crop.crop((0, ry1, crop_w, ry2))
        raw = pytesseract.image_to_string(_preprocess_row(row), config=_LOOT_OCR_CONFIG).strip()
        # Strip spaces and commas (thousands separators vary)
        raw = raw.replace(",", "").replace(" ", "")
        result[name] = int(raw) if raw.isdigit() else 0
    return result


def analyze_screenshot_with_ocr(screenshot: Image.Image, bounds: tuple = _RESOURCE_BOUNDS) -> dict:
    """
    Extract loot values from a CoC scout screenshot using OCR on a
    fixed coordinate region — no AI or network call required.
    Returns: {"gold": int, "elixir": int, "dark_elixir": int}
    """
    x1, y1, x2, y2 = bounds
    crop = screenshot.crop((x1, y1, x2, y2))
    crop_w, crop_h = crop.size

    row_h = (crop_h - _ROW_TOP) // 3
    result = {}
    for i, name in enumerate(("gold", "elixir", "dark_elixir")):
        ry1 = _ROW_TOP + i * row_h
        ry2 = _ROW_TOP + (i + 1) * row_h
        row = crop.crop((_ICON_WIDTH, ry1, crop_w, ry2))
        raw = pytesseract.image_to_string(_preprocess_row(row), config=_OCR_CONFIG).strip()
        result[name] = int(raw) if raw.isdigit() else 0

    return result

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = "us-east-1"
BEDROCK_MODEL_ID      = "us.amazon.nova-lite-v1:0"
SCREENSHOT_REGION     = None   # None = full screen, or (x, y, w, h)
SAMPLE_IMAGE_DIR      = "image/sample"

# ──────────────────────────────────────────────
# BEDROCK CLIENT
# ──────────────────────────────────────────────
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def analyze_screenshot_with_bedrock(screenshot: Image.Image, bounds: tuple = _RESOURCE_BOUNDS) -> dict:
    """
    Send a cropped loot region to AWS Bedrock Nova Lite and extract loot values.
    Returns: {"gold": int, "elixir": int, "dark_elixir": int}
    """
    x1, y1, x2, y2 = bounds
    crop = screenshot.crop((x1, y1, x2, y2))
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = (
        "You are analyzing a Clash of Clans screenshot of an opponent's base "
        "during the matchmaking/scouting screen. "
        "Extract the available loot: Gold, Elixir, and Dark Elixir (if visible, should be on the left side somewhere)."
        "Return ONLY a JSON object with keys: gold, elixir, dark_elixir (integers, 0 if not visible). "
        'Example: {"gold": 1500000, "elixir": 1200000, "dark_elixir": 8000}'
    )

    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": img_b64},
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "inferenceConfig": {
            "max_new_tokens": 256,
            "temperature": 0.1,
        },
    })

    response = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
    result_body = json.loads(response["body"].read())
    raw_text = result_body["output"]["message"]["content"][0]["text"].strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"gold": 0, "elixir": 0, "dark_elixir": 0}