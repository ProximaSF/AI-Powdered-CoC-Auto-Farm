import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
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