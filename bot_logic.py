import json
import math
import time
import os
import glob
import threading
import importlib
import numpy as np
import cv2
import pyautogui
from PIL import Image
from pynput.keyboard import Controller as KeyboardController

_keyboard = KeyboardController()

from analyze_screenshot import (analyze_screenshot_with_bedrock, analyze_screenshot_with_ocr,
                                 analyze_user_resources_with_ai, validate_storage_full_with_ai,
                                 read_looted_resources_ocr)
from discord_notify import webhook_embed

SCREENSHOT_REGION  = None   # None = full screen, or (x, y, w, h)
COORDS_FILE        = "coc_button_coords.json"
GLOBAL_CONFIG      = "global.json"
RETURN_HOME_IMAGE     = "image/detect_image/return_home_button.png"
FIND_NEXT_BATTLE_IMAGE = "image/detect_image/find_next_battle.png"
BATTLE_DURATION    = 180    # seconds
AIR_DEFENCE_GLOB = "image/detect_image/air_defences_*.png"

IGNORE_BOUNDRY = ["1481,1443", "851,1417"] # rectangle to prevent troop placement over troop menu



def _load_coords():
    with open(COORDS_FILE, "r") as f:
        data = json.load(f)

    def parse(s):
        x, y = s.split(",")
        return int(x.strip()), int(y.strip())

    diamond = data["troop_deployment_outer_diamond_bountry"]

    opp = data["opponent_resource_boundry"]
    otl = parse(opp["top_left"])
    obr = parse(opp["bottom_right"])

    ur = data["user_resource_boundry"]
    tl = parse(ur["top_left"])
    br = parse(ur["bottom_right"])

    lr = data["resource_looted_boundry"]
    ltl = parse(lr["top_left"])
    lbr = parse(lr["bottom_right"])

    return {
        "attack":                  parse(data["home_to_battle"]["attack_button"]),
        "find_match":              parse(data["home_to_battle"]["find_a_match_button"]),
        "final_attack":            parse(data["home_to_battle"]["final_attack_button"]),
        "next_battle":             parse(data["next_battle"]["next_battle_button"]),
        "return_home_from_battle": parse(data["return_home_from_battle"]["return_home_button"]),
        "surrender":               parse(data["surrender"]["surrender_button"]),
        "surrender_next":          parse(data["surrender"]["next_button"]),
        "surrender_home":          parse(data["surrender"]["return_home_button"]),
        "opponent_resource_bounds": (otl[0], otl[1], obr[0], obr[1]),
        "user_resource_bounds":    (tl[0], tl[1], br[0], br[1]),
        "looted_resource_bounds":  (ltl[0], ltl[1], lbr[0], lbr[1]),
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


def _find_all_template_matches(screenshot, template_path, threshold=0.75):
    """Find all non-overlapping instances of template in screenshot.
    Returns list of (x, y) center screen coordinates."""
    needle = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if needle is None:
        return []
    h, w = needle.shape
    result = cv2.matchTemplate(
        cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY),
        needle,
        cv2.TM_CCOEFF_NORMED,
    )
    centers = []
    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            break
        cx = max_loc[0] + w // 2
        cy = max_loc[1] + h // 2
        centers.append((cx, cy))
        # Suppress region around this match to avoid duplicates
        x1 = max(0, max_loc[0] - w // 2)
        y1 = max(0, max_loc[1] - h // 2)
        x2 = min(result.shape[1], max_loc[0] + w * 3 // 2)
        y2 = min(result.shape[0], max_loc[1] + h * 3 // 2)
        result[y1:y2, x1:x2] = 0.0
    return centers


def _find_air_defences(screenshot, threshold=0.65, dedup_dist=60):
    """Search all air defence templates and return deduplicated
    list of (x, y) center coordinates for each air defence found."""
    raw = []
    for path in sorted(glob.glob(AIR_DEFENCE_GLOB)):
        raw.extend(_find_all_template_matches(screenshot, path, threshold))

    # Deduplicate: if two hits are within dedup_dist pixels, keep only one
    unique = []
    for cx, cy in raw:
        if not any(abs(cx - ux) < dedup_dist and abs(cy - uy) < dedup_dist
                   for ux, uy in unique):
            unique.append((cx, cy))
    return unique


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

        self.min_combined_loot   = _g["min_combined_loot"]["amount"]
        self.check_combined      = _g["min_combined_loot"]["active"]

        self.max_gold_storage        = _g.get("max_user_gold_resource_storage", 0)
        self.max_elixir_storage      = _g.get("max_user_elixir_resource_storage", 0)
        self.max_dark_elixir_storage = _g.get("max_user_dark_elixir_resource_storage", 0)

        self.test_mode       = False
        self.test_image_path = None
        self.use_ocr         = True   # default: fast OCR, no AI

        self.selected_army   = "army_1"
        self._next_battled   = False

        # Running resource counters (set at startup, incremented after each battle)
        self._user_gold        = 0
        self._user_elixir      = 0
        self._user_dark_elixir = 0

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

    def update_thresholds(self, gold, elixir, dark_elixir, combined_loot,
                          check_gold, check_elixir, check_dark_elixir, check_combined):
        self.min_gold            = gold
        self.min_elixir          = elixir
        self.min_dark_elixir     = dark_elixir
        self.min_combined_loot   = combined_loot
        self.check_gold          = check_gold
        self.check_elixir        = check_elixir
        self.check_dark_elixir   = check_dark_elixir
        self.check_combined      = check_combined

    def _loop(self):
        self.log("Bot started.")
        try:
            coords = _load_coords()
        except Exception as e:
            self.log(f"[ERROR] Failed to load button coords: {e}")
            self.stop()
            return

        if not self.test_mode:
            # Send startup Discord ping to confirm webhook is working
            try:
                webhook_embed("CoC Farm Bot — Started", "Bot has started farming.")
                self.log("[DISCORD] Startup notification sent.")
            except Exception as ex:
                self.log(f"[DISCORD] Startup notification failed: {ex}")

            self.log("[STORAGE] Reading starting resources with AI...")
            self.log(f"[STORAGE] Max limits — Gold: {self.max_gold_storage:,}  "
                     f"Elixir: {self.max_elixir_storage:,}  "
                     f"Dark Elixir: {self.max_dark_elixir_storage:,}")
            try:
                start_ss = self._take_screenshot()
                start_res = analyze_user_resources_with_ai(start_ss, coords["user_resource_bounds"])
                self._user_gold        = start_res.get("gold", 0)
                self._user_elixir      = start_res.get("elixir", 0)
                self._user_dark_elixir = start_res.get("dark_elixir", 0)
                self.log(f"[STORAGE] Starting — Gold: {self._user_gold:,}  "
                         f"Elixir: {self._user_elixir:,}  "
                         f"Dark Elixir: {self._user_dark_elixir:,}")

                # Check if already full before doing a single battle.
                # Only consider resources that are selected in the GUI.
                # Stop only when ALL selected resources are full.
                already_full = []
                if self.check_gold and self.max_gold_storage > 0 and self._user_gold >= self.max_gold_storage:
                    already_full.append("Gold")
                if self.check_elixir and self.max_elixir_storage > 0 and self._user_elixir >= self.max_elixir_storage:
                    already_full.append("Elixir")
                if self.check_dark_elixir and self.max_dark_elixir_storage > 0 and self._user_dark_elixir >= self.max_dark_elixir_storage:
                    already_full.append("Dark Elixir")

                selected_count = sum([self.check_gold, self.check_elixir, self.check_dark_elixir])
                if already_full:
                    self.log(f"[STORAGE] Full at startup: {', '.join(already_full)} "
                             f"({len(already_full)}/{selected_count} selected resources)")
                if already_full and len(already_full) >= selected_count:
                    msg = "All selected storage full at startup: " + ", ".join(already_full)
                    self.log(f"[STORAGE FULL] {msg}. Not starting battles.")
                    try:
                        webhook_embed("CoC Farm Bot — Storage Full", msg)
                    except Exception as ex:
                        self.log(f"[DISCORD] Notification failed: {ex}")
                    self.stop()
                    return
            except Exception as ex:
                self.log(f"[STORAGE] Could not read starting resources: {ex}")

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

        if self.use_ocr:
            self.log("Analyzing resources with OCR...")
            resources = analyze_screenshot_with_ocr(screenshot, coords["opponent_resource_bounds"])
        else:
            self.log("Analyzing resources with Bedrock AI...")
            resources = analyze_screenshot_with_bedrock(screenshot, coords["opponent_resource_bounds"])
        gold        = resources.get("gold", 0)
        elixir      = resources.get("elixir", 0)
        dark_elixir = resources.get("dark_elixir", 0)

        parts = []
        if self.check_gold:        parts.append(f"Gold: {gold:,}")
        if self.check_elixir:      parts.append(f"Elixir: {elixir:,}")
        if self.check_dark_elixir: parts.append(f"Dark Elixir: {dark_elixir:,}")
        self.log("Detected → " + "  |  ".join(parts) if parts else "Detected → (no resources selected)")

        # ── Dark Elixir: hard independent gate ──────────────────────────────
        dark_ok = (not self.check_dark_elixir) or (dark_elixir >= self.min_dark_elixir)
        if self.check_dark_elixir:
            self.log(f"  Dark Elixir: {dark_elixir:,} / {self.min_dark_elixir:,}  {'✅' if dark_ok else '❌ (required)'}")

        # ── Gold / Elixir: individual AND combined OR ────────────────────────
        loot_checks_active = self.check_gold or self.check_elixir or self.check_combined

        if self.check_gold:
            ind_gold_ok = gold >= self.min_gold
            self.log(f"  Gold:        {gold:,} / {self.min_gold:,}  {'✅' if ind_gold_ok else '❌'}")
        else:
            ind_gold_ok = True

        if self.check_elixir:
            ind_elixir_ok = elixir >= self.min_elixir
            self.log(f"  Elixir:      {elixir:,} / {self.min_elixir:,}  {'✅' if ind_elixir_ok else '❌'}")
        else:
            ind_elixir_ok = True

        # individual_ok: both active individual checks passed
        individual_ok = (ind_gold_ok and ind_elixir_ok) if (self.check_gold or self.check_elixir) else False

        # combined_ok: gold+elixir sum meets threshold
        combined_ok = False
        if self.check_combined:
            combined_sum = gold + elixir
            combined_ok = combined_sum >= self.min_combined_loot
            self.log(f"  Total G+E:   {combined_sum:,} / {self.min_combined_loot:,}  {'✅' if combined_ok else '❌'}")

        loot_ok = individual_ok or combined_ok

        if not loot_checks_active and not self.check_dark_elixir:
            self.log("[WARN] No resources selected to check — skipping base.")
            self._next_base(coords)
            if self.test_mode:
                self.log("\n[TEST] Single-cycle test complete. Bot stopped.")
                self.running = False
                self.set_status("Stopped")
            return

        if not dark_ok:
            self.log(f"Dark Elixir required but not met ({dark_elixir:,} / {self.min_dark_elixir:,}). Skipping...")
            self._next_base(coords)
            if self.test_mode:
                self.log("\n[TEST] Single-cycle test complete. Bot stopped.")
                self.running = False
                self.set_status("Stopped")
            return

        if loot_checks_active and not loot_ok:
            self.log("Loot below threshold. Moving to next battle...")
            self._next_base(coords)
            if self.test_mode:
                self.log("\n[TEST] Single-cycle test complete. Bot stopped.")
                self.running = False
                self.set_status("Stopped")
            return

        # Save original loot for early-surrender check during battle
        self._attack_resources = resources

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

        storage_full = self._wait_for_battle_end(coords, self._attack_resources)
        self._next_battled = False  # troops retrain after battle, order resets

        if self.running and not self.test_mode:
            if storage_full:
                self.stop()
                return
            self.log("Searching for next base...")
            time.sleep(1)
            self._navigate_to_battle(coords)

        if self.test_mode:
            self.log("\n[TEST] Single-cycle test complete. Bot stopped.")
            self.running = False
            self.set_status("Stopped")

    def _deploy_army(self, diamond, army):
        # Click the center of the deployment area to give the emulator keyboard focus.
        # With no troop type selected yet this click deploys nothing in CoC.
        cx = sum(p[0] for p in diamond) // 4
        cy = sum(p[1] for p in diamond) // 4
        pyautogui.click(cx, cy)
        time.sleep(0.3)

        try:
            strategy = importlib.import_module(f"army_strategies.{self.selected_army}")
        except ModuleNotFoundError:
            self.log(f"[ERROR] No strategy file found for '{self.selected_army}'. "
                     f"Create army_strategies/{self.selected_army}.py")
            return
        strategy.deploy(self, diamond, army)

    def _place_wave(self, key, points):
        import random
        self._press_key(key)
        for x, y in points:
            # Small position jitter ±4px so clicks aren't pixel-perfect
            jx = x + random.randint(-4, 4)
            jy = y + random.randint(-4, 4)
            pyautogui.click(jx, jy)
            # Human-like variable delay between troop placements
            time.sleep(random.uniform(0.08, 0.32))

    def _press_key(self, key):
        if len(key) != 1:
            self.log(f"[WARN] Invalid hotkey '{key}' — rename army entry to include single-char key suffix (e.g., dragon_1:)")
            return
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

    def _wait_for_battle_end(self, coords, original_resources: dict):
        _cfg = json.load(open(GLOBAL_CONFIG))
        check_after    = _cfg.get("surrender_check_after",    30)
        check_interval = _cfg.get("surrender_check_interval", 3)
        threshold      = _cfg.get("surrender_threshold",      0.25)

        self.log(f"[BATTLE] Waiting up to {BATTLE_DURATION}s for battle to end...")
        self.log(f"[BATTLE] Surrender check after {check_after}s, every {check_interval}s, threshold {threshold*100:.0f}%")
        start_time = time.time()
        deadline = start_time + BATTLE_DURATION
        needle = cv2.imread(RETURN_HOME_IMAGE, cv2.IMREAD_GRAYSCALE)
        last_loot_check = start_time  # so first check triggers after check_after seconds

        while time.time() < deadline and self.running:
            now = time.time()
            elapsed = now - start_time

            screenshot = self._take_screenshot()
            os.makedirs("image/ai_screenshot", exist_ok=True)
            screenshot.save("image/ai_screenshot/latest.png")

            # ── Early surrender check ────────────────────────────────────────
            if elapsed >= check_after and \
               now - last_loot_check >= check_interval:
                last_loot_check = now
                try:
                    current = analyze_screenshot_with_ocr(screenshot, coords["opponent_resource_bounds"]) if self.use_ocr \
                              else analyze_screenshot_with_bedrock(screenshot, coords["opponent_resource_bounds"])
                    triggers = []
                    if self.check_gold and original_resources.get("gold", 0) > 0:
                        ratio = current.get("gold", 0) / original_resources["gold"]
                        if ratio < threshold:
                            triggers.append(f"Gold {ratio*100:.0f}%")
                    if self.check_elixir and original_resources.get("elixir", 0) > 0:
                        ratio = current.get("elixir", 0) / original_resources["elixir"]
                        if ratio < threshold:
                            triggers.append(f"Elixir {ratio*100:.0f}%")
                    if self.check_dark_elixir and original_resources.get("dark_elixir", 0) > 0:
                        ratio = current.get("dark_elixir", 0) / original_resources["dark_elixir"]
                        if ratio < threshold:
                            triggers.append(f"Dark Elixir {ratio*100:.0f}%")

                    if triggers:
                        self.log(f"[BATTLE] Loot below {threshold*100:.0f}% ({', '.join(triggers)}). Surrendering early...")
                        return self._surrender(coords)
                    else:
                        self.log(f"[BATTLE] Loot check OK at {elapsed:.0f}s — continuing.")
                except Exception as e:
                    self.log(f"[BATTLE] Loot check error: {e}")

            # ── Check for natural battle end ─────────────────────────────────
            haystack = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val >= 0.8:
                self.log("[BATTLE] Return home button detected. Reading loot...")
                full = self._read_and_track_loot(coords, delay=0)
                self._click(coords["return_home_from_battle"])
                time.sleep(1)
                return full

            time.sleep(1)

        if self.running:
            self.log("[BATTLE] 3 minutes elapsed. Reading loot and returning home...")
            full = self._read_and_track_loot(coords, delay=0)
            self._click(coords["return_home_from_battle"])
            time.sleep(1)
            return full
        return False

    def _read_and_track_loot(self, coords, delay: float = 2.0) -> bool:
        """OCR the post-battle loot popup, add to running counters, and check if storage
        is full via AI. Returns True if AI confirms storage is full."""
        if delay > 0:
            time.sleep(delay)
        try:
            ss = self._take_screenshot()
            looted = read_looted_resources_ocr(ss, coords["looted_resource_bounds"])
        except Exception as e:
            self.log(f"[LOOT] OCR error: {e}")
            return False

        lg  = looted.get("gold", 0)
        le  = looted.get("elixir", 0)
        lde = looted.get("dark_elixir", 0)
        self.log(f"[LOOT] Looted this battle — Gold: {lg:,}  Elixir: {le:,}  Dark Elixir: {lde:,}")

        self._user_gold        += lg
        self._user_elixir      += le
        self._user_dark_elixir += lde
        self.log(f"[STORAGE] Running total — Gold: {self._user_gold:,}  "
                 f"Elixir: {self._user_elixir:,}  "
                 f"Dark Elixir: {self._user_dark_elixir:,}")

        gold_full   = self.check_gold        and self.max_gold_storage > 0        and self._user_gold        >= self.max_gold_storage
        elixir_full = self.check_elixir      and self.max_elixir_storage > 0      and self._user_elixir      >= self.max_elixir_storage
        de_full     = self.check_dark_elixir and self.max_dark_elixir_storage > 0 and self._user_dark_elixir >= self.max_dark_elixir_storage

        selected_count = sum([self.check_gold, self.check_elixir, self.check_dark_elixir])
        full_selected  = sum([gold_full, elixir_full, de_full])

        if full_selected > 0:
            partial = ([" Gold"] if gold_full else []) + (["Elixir"] if elixir_full else []) + (["Dark Elixir"] if de_full else [])
            self.log(f"[STORAGE] Full so far: {', '.join(partial)} ({full_selected}/{selected_count} selected resources)")

        # Only trigger AI validation when ALL selected resources are full
        if full_selected < selected_count:
            return False

        suspects = ([" Gold"] if gold_full else []) + (["Elixir"] if elixir_full else []) + (["Dark Elixir"] if de_full else [])
        self.log(f"[STORAGE] Counter suggests {', '.join(suspects)} may be full — asking AI to validate...")

        try:
            home_ss    = self._take_screenshot()
            validation = validate_storage_full_with_ai(home_ss, coords["user_resource_bounds"])
        except Exception as e:
            self.log(f"[STORAGE] AI validation error: {e}")
            return False

        confirmed = []
        if gold_full   and validation.get("gold_full"):        confirmed.append("Gold")
        if elixir_full and validation.get("elixir_full"):      confirmed.append("Elixir")
        if de_full     and validation.get("dark_elixir_full"): confirmed.append("Dark Elixir")

        if confirmed:
            msg = "Storage full: " + ", ".join(confirmed)
            self.log(f"[STORAGE FULL] AI confirmed — {msg}. Stopping bot.")
            try:
                webhook_embed("CoC Farm Bot — Storage Full", msg)
            except Exception as e:
                self.log(f"[DISCORD] Notification failed: {e}")
            return True

        self.log("[STORAGE] AI says storage not full yet — continuing.")
        return False

    def _surrender(self, coords) -> bool:
        """Click surrender, read loot popup, then return home.
        Returns True if AI confirms storage is full."""
        self._click(coords["surrender"])
        time.sleep(1)
        self._click(coords["surrender_next"])
        # Loot popup appears after next_button — read it (delay built into _read_and_track_loot)
        full = self._read_and_track_loot(coords)
        self._click(coords["surrender_home"])
        time.sleep(1)
        return full

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
