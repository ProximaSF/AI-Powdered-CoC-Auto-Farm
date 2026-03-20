import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import time
import os
import pyautogui
from PIL import Image
from pynput import keyboard as pynput_keyboard
from dotenv import load_dotenv
load_dotenv()

from analyze_screenshot import analyze_screenshot_with_bedrock

SCREENSHOT_REGION     = None   # None = full screen, or (x, y, w, h)


class CoCFarmBot:
    def __init__(self, log_callback, status_callback):
        self.running      = False
        self.log          = log_callback
        self.set_status   = status_callback

        # Thresholds
        self.min_gold        = 500000
        self.min_elixir      = 500000
        self.min_dark_elixir = 9000

        # Which resources to check
        self.check_gold        = True
        self.check_elixir      = True
        self.check_dark_elixir = True

        # Test mode
        self.test_mode       = False
        self.test_image_path = None

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

    def update_thresholds(self, gold, elixir, dark_elixir,
                          check_gold, check_elixir, check_dark_elixir):
        self.min_gold        = gold
        self.min_elixir      = elixir
        self.min_dark_elixir = dark_elixir
        self.check_gold        = check_gold
        self.check_elixir      = check_elixir
        self.check_dark_elixir = check_dark_elixir

    def _loop(self):
        self.log("Bot started. Scanning for bases...")
        cycle = 0
        while self.running:
            cycle += 1
            self.log(f"\n── Cycle {cycle} ──")
            try:
                self._run_cycle()
            except Exception as e:
                self.log(f"[ERROR] {e}")
                time.sleep(3)
        self.log("Bot loop exited.")

    def _run_cycle(self):
        # 1. Take / load screenshot
        if self.test_mode and self.test_image_path:
            self.log(f"[TEST] Loading image: {os.path.basename(self.test_image_path)}")
            screenshot = Image.open(self.test_image_path).convert("RGB")
        else:
            self.log("Taking screenshot...")
            screenshot = self._take_screenshot()

        # 2. Analyze with Bedrock
        self.log("Analyzing resources with Bedrock AI...")
        resources   = analyze_screenshot_with_bedrock(screenshot)
        gold        = resources.get("gold", 0)
        elixir      = resources.get("elixir", 0)
        dark_elixir = resources.get("dark_elixir", 0)

        # Log only the resources being checked
        parts = []
        if self.check_gold:        parts.append(f"Gold: {gold:,}")
        if self.check_elixir:      parts.append(f"Elixir: {elixir:,}")
        if self.check_dark_elixir: parts.append(f"Dark Elixir: {dark_elixir:,}")
        self.log("Detected → " + "  |  ".join(parts) if parts else "Detected → (no resources selected)")

        # 3. Check only the enabled thresholds
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
            self._next_base()
            return

        if all(results):
            self.log("All thresholds met! Attacking base...")
            self._attack()
        else:
            self.log("Below threshold. Skipping to next base...")
            self._next_base()

        # Stop after one cycle in test mode
        if self.test_mode:
            self.log("\n[TEST] Single-cycle test complete. Bot stopped.")
            self.running = False
            self.set_status("Stopped")

        time.sleep(2)

    def _take_screenshot(self) -> Image.Image:
        if SCREENSHOT_REGION:
            x, y, w, h = SCREENSHOT_REGION
            return pyautogui.screenshot(region=(x, y, w, h))
        return pyautogui.screenshot()

    def _attack(self):
        self.log("[ATTACK] → (AI troop placement coming in Phase 2)")
        time.sleep(5)

    def _next_base(self):
        self.log("[SKIP]   → Moving to next base... (click logic in Phase 2)")
        time.sleep(2)
