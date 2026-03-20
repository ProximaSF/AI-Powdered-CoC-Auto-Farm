import json
import math
import time
import os
import threading
import importlib
import numpy as np
import cv2
import pyautogui
from PIL import Image
from pynput.keyboard import Controller as KeyboardController

_keyboard = KeyboardController()

from analyze_screenshot import analyze_screenshot_with_bedrock

SCREENSHOT_REGION  = None   # None = full screen, or (x, y, w, h)
COORDS_FILE        = "coc_button_coords.json"
GLOBAL_CONFIG      = "global.json"
RETURN_HOME_IMAGE     = "image/detect_image/return_home_button.png"
FIND_NEXT_BATTLE_IMAGE = "image/detect_image/find_next_battle.png"
BATTLE_DURATION    = 180    # seconds

IGNORE_BOUNDRY = ["1043,1435", "1498,1420"] # rectangle to prevent troop placement over troop menu



def _load_coords():
    with open(COORDS_FILE, "r") as f:
        data = json.load(f)

    def parse(s):
        x, y = s.split(",")
        return int(x.strip()), int(y.strip())

    diamond = data["troop_deployment_outer_diamond_bountry"]

    return {
        "attack":                  parse(data["home_to_battle"]["attack_button"]),
        "find_match":              parse(data["home_to_battle"]["find_a_match_button"]),
        "final_attack":            parse(data["home_to_battle"]["final_attack_button"]),
        "next_battle":             parse(data["next_battle"]["next_battle_button"]),
        "return_home_from_battle": parse(data["return_home_from_battle"]["return_home_button"]),
        "diamond": [
            parse(diamond["left"]),
            parse(diamond["top"]),
            parse(diamond["right"]),
            parse(diamond["bottom"]),
        ],
    }


def _load_army(army_key="army_1"):
    """Load the static army config from army_strategies/{army_key}.json."""
    path = os.path.join("army_strategies", f"{army_key}.json")
    with open(path, "r") as f:
        data = json.load(f)

    army = {}
    for raw_key, count in data.items():
        name = raw_key.rstrip(":")
        press_key = name.split("_")[-1]

        if name.startswith("hero") or name.startswith("her0"):
            category = "hero"
        elif "wall_breaker" in name:
            category = "wall_breaker"
        elif "spell" in name:
            category = "spell"
        elif "healer" in name:
            category = "healer"
        else:
            category = "troop"

        army[name] = {"key": press_key, "count": count, "category": category}

    return army


def _parse_ignore_boundary():
    coords = [tuple(int(v.strip()) for v in s.split(",")) for s in IGNORE_BOUNDRY]
    x1 = min(c[0] for c in coords)
    x2 = max(c[0] for c in coords)
    y1 = min(c[1] for c in coords)
    return x1, y1, x2


_IGNORE_X1, _IGNORE_Y1, _IGNORE_X2 = _parse_ignore_boundary()


def _in_ignore_zone(x, y):
    return _IGNORE_X1 <= x <= _IGNORE_X2 and y >= _IGNORE_Y1


def _perimeter_points(diamond, num_points):
    """Evenly spaced points along diamond perimeter, skipping ignore zone.
    Oversamples so the ignore zone never reduces the returned count."""
    edges = []
    total_length = 0
    for i in range(4):
        p1 = diamond[i]
        p2 = diamond[(i + 1) % 4]
        length = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        edges.append((p1, p2, length))
        total_length += length

    # Generate 10× candidates so the ignore zone never starves us of points
    oversample = num_points * 10
    candidates = []
    for i in range(oversample):
        target = (i / oversample) * total_length
        remaining = target
        for p1, p2, length in edges:
            if remaining <= length:
                frac = remaining / length
                x = int(p1[0] + frac * (p2[0] - p1[0]))
                y = int(p1[1] + frac * (p2[1] - p1[1]))
                if not _in_ignore_zone(x, y):
                    candidates.append((x, y))
                break
            remaining -= length

    if len(candidates) <= num_points:
        return candidates

    # Evenly subsample from the valid candidates
    step = len(candidates) / num_points
    return [candidates[int(i * step)] for i in range(num_points)]


def _inward_points(diamond, num_points, inward=250):
    """Like perimeter_points but shifted toward the center by `inward` pixels."""
    cx = sum(p[0] for p in diamond) / 4
    cy = sum(p[1] for p in diamond) / 4
    pts = _perimeter_points(diamond, num_points)
    result = []
    for x, y in pts:
        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            result.append((int(x + inward * dx / dist), int(y + inward * dy / dist)))
    return result


class CoCFarmBot:
    def __init__(self, log_callback, status_callback):
        self.running      = False
        self.log          = log_callback
        self.set_status   = status_callback

        _g = json.load(open(GLOBAL_CONFIG))
        self.min_gold        = _g["min_gold"]["amount"]
        self.min_elixir      = _g["min_elixir"]["amount"]
        self.min_dark_elixir = _g["min_dark_elixir"]["amount"]

        self.check_gold        = _g["min_gold"]["active"]
        self.check_elixir      = _g["min_elixir"]["active"]
        self.check_dark_elixir = _g["min_dark_elixir"]["active"]

        self.test_mode       = False
        self.test_image_path = None

        self.selected_army   = "army_1"
        self._next_battled   = False

        self._thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.set_status("Running")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.set_status("Stopped")
        self.log("Bot stopped.")

    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.start()

    def update_army(self, army_key: str):
        self.selected_army = army_key

    def update_thresholds(self, gold, elixir, dark_elixir,
                          check_gold, check_elixir, check_dark_elixir):
        self.min_gold        = gold
        self.min_elixir      = elixir
        self.min_dark_elixir = dark_elixir
        self.check_gold        = check_gold
        self.check_elixir      = check_elixir
        self.check_dark_elixir = check_dark_elixir

    def _loop(self):
        self.log("Bot started.")
        try:
            coords = _load_coords()
        except Exception as e:
            self.log(f"[ERROR] Failed to load button coords: {e}")
            self.stop()
            return

        if not self.test_mode:
            self._navigate_to_battle(coords)

        cycle = 0
        while self.running:
            cycle += 1
            self.log(f"\n── Cycle {cycle} ──")
            try:
                self._run_cycle(coords)
            except Exception as e:
                self.log(f"[ERROR] {e}")
                time.sleep(3)

        self.log("Bot loop exited.")

    def _navigate_to_battle(self, coords):
        self.log("Navigating to battle from home village...")
        self._click(coords["attack"])
        self.log("  Clicked Attack button. Waiting...")
        time.sleep(0.5)

        self._click(coords["find_match"])
        self.log("  Clicked Find a Match. Waiting for matchmaking...")
        time.sleep(0.5)

        self._click(coords["final_attack"])
        self.log("  Clicked final Attack. Waiting for scout screen...")
        time.sleep(2)
        if not self._wait_for_image(FIND_NEXT_BATTLE_IMAGE, timeout=30):
            self.log("  [WARN] Scout screen not detected within 30s, proceeding anyway.")

    def _run_cycle(self, coords):
        if self.test_mode and self.test_image_path:
            self.log(f"[TEST] Loading image: {os.path.basename(self.test_image_path)}")
            screenshot = Image.open(self.test_image_path).convert("RGB")
        else:
            self.log("Taking screenshot...")
            screenshot = self._take_screenshot()
            os.makedirs("image/ai_screenshot", exist_ok=True)
            screenshot.save("image/ai_screenshot/latest.png")

        self.log("Analyzing resources with Bedrock AI...")
        resources   = analyze_screenshot_with_bedrock(screenshot)
        gold        = resources.get("gold", 0)
        elixir      = resources.get("elixir", 0)
        dark_elixir = resources.get("dark_elixir", 0)

        parts = []
        if self.check_gold:        parts.append(f"Gold: {gold:,}")
        if self.check_elixir:      parts.append(f"Elixir: {elixir:,}")
        if self.check_dark_elixir: parts.append(f"Dark Elixir: {dark_elixir:,}")
        self.log("Detected → " + "  |  ".join(parts) if parts else "Detected → (no resources selected)")

        results = []
        if self.check_gold:
            ok = gold >= self.min_gold
            results.append(ok)
            self.log(f"  Gold:        {gold:,} / {self.min_gold:,}  {'✅' if ok else '❌'}")
        if self.check_elixir:
            ok = elixir >= self.min_elixir
            results.append(ok)
            self.log(f"  Elixir:      {elixir:,} / {self.min_elixir:,}  {'✅' if ok else '❌'}")
        if self.check_dark_elixir:
            ok = dark_elixir >= self.min_dark_elixir
            results.append(ok)
            self.log(f"  Dark Elixir: {dark_elixir:,} / {self.min_dark_elixir:,}  {'✅' if ok else '❌'}")

        if not results:
            self.log("[WARN] No resources selected to check — skipping base.")
            self._next_base(coords)
            return

        if not all(results):
            self.log("Below threshold. Moving to next battle...")
            self._next_base(coords)
            if self.test_mode:
                self.log("\n[TEST] Single-cycle test complete. Bot stopped.")
                self.running = False
                self.set_status("Stopped")
            return

        # Thresholds met — attack
        # Use reorder variant if troops were reshuffled by a previous "Next Battle"
        reorder_key = f"{self.selected_army}_reorder"
        reorder_path = os.path.join("army_strategies", f"{reorder_key}.json")
        if self._next_battled and os.path.exists(reorder_path):
            army_key = reorder_key
            self.log(f"All thresholds met! Deploying army ({army_key}.json — reorder variant)...")
        else:
            army_key = self.selected_army
            self.log(f"All thresholds met! Deploying army ({army_key}.json)...")
        army = _load_army(army_key)
        self._deploy_army(coords["diamond"], army)

        self._wait_for_battle_end(coords)
        self._next_battled = False  # troops retrain after battle, order resets

        if self.running and not self.test_mode:
            self.log("Returning to home village and searching for next base...")
            time.sleep(1)
            self._navigate_to_battle(coords)

        if self.test_mode:
            self.log("\n[TEST] Single-cycle test complete. Bot stopped.")
            self.running = False
            self.set_status("Stopped")

    def _deploy_army(self, diamond, army):
        try:
            strategy = importlib.import_module(f"army_strategies.{self.selected_army}")
        except ModuleNotFoundError:
            self.log(f"[ERROR] No strategy file found for '{self.selected_army}'. "
                     f"Create army_strategies/{self.selected_army}.py")
            return
        strategy.deploy(self, diamond, army)

    def _place_wave(self, key, points):
        self._press_key(key)
        for x, y in points:
            pyautogui.click(x, y)
            time.sleep(0.05)

    def _press_key(self, key):
        _keyboard.press(key)
        _keyboard.release(key)
        time.sleep(0.2)

    def _wait_for_image(self, image_path, timeout=60, threshold=0.8, poll=1.0):
        """Poll screenshots until image_path is found or timeout expires.
        Returns True if found, False on timeout."""
        needle = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        deadline = time.time() + timeout
        while time.time() < deadline and self.running:
            screenshot = self._take_screenshot()
            haystack = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= threshold:
                return True
            time.sleep(poll)
        return False

    def _wait_for_battle_end(self, coords):
        self.log(f"[BATTLE] Waiting up to {BATTLE_DURATION}s for battle to end...")
        deadline = time.time() + BATTLE_DURATION
        needle = cv2.imread(RETURN_HOME_IMAGE, cv2.IMREAD_GRAYSCALE)

        while time.time() < deadline and self.running:
            screenshot = self._take_screenshot()
            os.makedirs("image/ai_screenshot", exist_ok=True)
            screenshot.save("image/ai_screenshot/latest.png")

            haystack = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            self.log(f"[BATTLE] Return home match score: {max_val:.2f}")

            if max_val >= 0.8:
                self.log("[BATTLE] Return home button detected early! Clicking...")
                self._click(coords["return_home_from_battle"])
                time.sleep(1)
                return

            time.sleep(1)

        if self.running:
            self.log("[BATTLE] 3 minutes elapsed. Clicking return home...")
            self._click(coords["return_home_from_battle"])
            time.sleep(1)

    def _take_screenshot(self) -> Image.Image:
        if SCREENSHOT_REGION:
            x, y, w, h = SCREENSHOT_REGION
            return pyautogui.screenshot(region=(x, y, w, h))
        return pyautogui.screenshot()

    def _next_base(self, coords):
        self.log("[SKIP] Clicking Next Battle...")
        self._next_battled = True
        self._click(coords["next_battle"])
        self.log("[SKIP] Waiting for next scout screen...")
        time.sleep(2)
        if not self._wait_for_image(FIND_NEXT_BATTLE_IMAGE, timeout=30):
            self.log("[SKIP] [WARN] Scout screen not detected within 30s, proceeding anyway.")
        else:
            time.sleep(1)  # let the new base fully render before the next screenshot

    def _click(self, coord):
        x, y = coord
        pyautogui.click(x, y)
