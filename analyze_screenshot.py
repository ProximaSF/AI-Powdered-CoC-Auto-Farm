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


def analyze_screenshot_with_ocr(screenshot: Image.Image) -> dict:
    """
    Extract loot values from a CoC scout screenshot using OCR on a
    fixed coordinate region — no AI or network call required.
    Returns: {"gold": int, "elixir": int, "dark_elixir": int}
    """
    x1, y1, x2, y2 = _RESOURCE_BOUNDS
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


def analyze_screenshot_with_bedrock(screenshot: Image.Image) -> dict:
    """
    Send a screenshot to AWS Bedrock Nova Lite and extract loot values.
    Returns: {"gold": int, "elixir": int, "dark_elixir": int}
    """
    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
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