# Clash of Clans Auto Farm Bot

All vib coded

---

## Requirements

- Python 3.10+
- AWS account with Bedrock access (Nova Lite model enabled in `us-east-1`)
- Clash of Clans running on a screen/emulator

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
```

### 2. Folder Structure

Create the following folders before running the bot:

```
image/
├── detect_image/       # Required — OpenCV template images for screen detection
│   ├── find_next_battle.png      # Screenshot of the "Next" button on the scout screen
│   └── return_home_button.png    # Screenshot of the "Return Home" button post-battle
├── sample/             # Optional — test screenshots for Test Mode
└── ai_screenshot/      # Auto-created by the bot — stores latest live screenshot
```

**detect_image** images must be cropped screenshots of the exact UI elements as they appear on your screen/emulator. The bot uses OpenCV template matching to detect them (match threshold: 0.8).

### 3. Button Coordinates

Edit `coc_button_coords.json` to match your screen resolution and emulator layout:

```json
{
    "troop_deployment_outer_diamond_bountry": {
        "left":   "x,y",
        "top":    "x,y",
        "right":  "x,y",
        "bottom": "x,y"
    },
    "home_to_battle": {
        "attack_button":       "x,y",
        "find_a_match_button": "x,y",
        "final_attack_button": "x,y"
    },
    "next_battle": {
        "next_battle_button": "x,y"
    },
    "return_home_from_battle": {
        "return_home_button": "x,y"
    }
}
```

### 4. Army Strategies

Army configs live in `army_strategies/`. Each army needs two files:

- `army_1.json` — troop counts and keybinds (normal order)
- `army_1_reorder.json` — same army after pressing "Next Battle" (troop bar order may shift)
- `army_1.py` — deployment logic called by the bot

Key naming format in JSON: `category_name_KEY` where the last segment is the in-game hotkey.

### 5. Global Defaults

Edit `global.json` to set default resource thresholds and which are checked on startup:

```json
{
    "min_gold":        {"amount": 500000, "active": true},
    "min_elixir":      {"amount": 500000, "active": true},
    "min_dark_elixir": {"amount": 5000,   "active": false}
}
```

---

## Running

```bash
python main.py
```

- **F8** — global hotkey to start/stop the bot at any time
- Use **Test Mode** in the GUI to analyze a sample image without live automation

---

## File Overview

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `bot_gui.py` | Tkinter GUI |
| `bot_logic.py` | Core bot loop and automation logic |
| `analyze_screenshot.py` | AWS Bedrock call for resource detection |
| `global.json` | Default thresholds and active flags |
| `coc_button_coords.json` | Screen coordinates for all clickable buttons |
| `army_strategies/army_N.json` | Troop counts and keybinds for army N |
| `army_strategies/army_N.py` | Deployment strategy for army N |
