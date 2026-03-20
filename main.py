"""
CoC Farm Bot - Phase 1
Features:
  - Tkinter GUI with Start/Stop controls
  - Global keybind (F8) to toggle bot
  - Resource threshold settings with per-resource enable/disable checkboxes
  - Test mode: load screenshot from image/sample folder instead of live capture
  - AWS Bedrock (Nova Lite) screenshot analysis to read opponent resources
  - Decision logic: attack if selected resources >= threshold, else skip
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
import time
import os
import json
import base64
import io
import pyautogui
import boto3
from PIL import Image
from pynput import keyboard as pynput_keyboard
from dotenv import load_dotenv
load_dotenv()

TOGGLE_HOTKEY = pynput_keyboard.Key.f8

from bot_gui import BotGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()