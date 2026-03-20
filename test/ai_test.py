import json
import base64
import io
import boto3
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

AWS_REGION       = "us-east-1"
BEDROCK_MODEL_ID = "us.amazon.nova-2-lite-v1:0"
IMAGE_PATH       = "image/sample/screenshot1.png"

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
 
# Load & encode image
img = Image.open(IMAGE_PATH).convert("RGB")
buffer = io.BytesIO()
img.save(buffer, format="PNG")
img_bytes = buffer.getvalue()   # raw bytes — NOT base64 string for Nova
 
prompt = (
    "You are analyzing a Clash of Clans screenshot of an opponent's base "
    "during the matchmaking/scouting screen. "
    "Extract the available loot: Gold, Elixir, and Dark Elixir (if visible). "
    "Return ONLY a JSON object with keys: gold, elixir, dark_elixir (integers, 0 if not visible). "
    'Example: {"gold": 1500000, "elixir": 1200000, "dark_elixir": 8000}'
)
 
# Nova Lite format — NOTE: image bytes go in directly, NOT base64 encoded
body = json.dumps({
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {
                            "bytes": base64.b64encode(img_bytes).decode("utf-8")
                        },
                    }
                },
                {
                    "text": prompt
                },
            ],
        }
    ],
    "inferenceConfig": {
        "max_new_tokens": 256,
        "temperature": 0.1,
    },
})
 
print(f"Sending image to Bedrock ({BEDROCK_MODEL_ID})...")
response = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
result_body = json.loads(response["body"].read())
 
# Nova response structure is different from Claude
# {"output": {"message": {"content": [{"text": "..."}]}}}
raw_text = result_body["output"]["message"]["content"][0]["text"].strip()
raw_text = raw_text.replace("```json", "").replace("```", "").strip()
 
print(f"\nRaw response:\n{raw_text}\n")
 
try:
    data = json.loads(raw_text)
    print(f"Gold:        {data.get('gold', 0):,}")
    print(f"Elixir:      {data.get('elixir', 0):,}")
    print(f"Dark Elixir: {data.get('dark_elixir', 0):,}")
except json.JSONDecodeError:
    print("Could not parse JSON from response.")
