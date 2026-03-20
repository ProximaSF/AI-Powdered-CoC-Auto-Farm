import os
import json
import base64
import io
import boto3
from PIL import Image
from pynput import keyboard as pynput_keyboard
from dotenv import load_dotenv
load_dotenv()

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