import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import os
import json
from PIL import Image
from pynput import keyboard as pynput_keyboard

from bot_logic import CoCFarmBot

from analyze_screenshot import analyze_screenshot_with_bedrock

SAMPLE_IMAGE_DIR = "image/sample"
GLOBAL_CONFIG    = "global.json"

def _load_global():
    with open(GLOBAL_CONFIG, "r") as f:
        return json.load(f)

_globals = _load_global()

class BotGUI():
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("CoC Farm Bot")
        root.resizable(False, False)
        root.configure(bg="#1e1e2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel",      background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TFrame",      background="#1e1e2e")
        style.configure("TCheckbutton",background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#1e1e2e", foreground="#89b4fa")
        style.configure("TLabelframe.Label", background="#1e1e2e", foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        style.configure("TEntry",      fieldbackground="#313244", foreground="#cdd6f4")
        style.map("TCheckbutton", background=[("active", "#1e1e2e")])

        self.bot = CoCFarmBot(
            log_callback=self._log,
            status_callback=self._set_status,
        )

        self._build_ui()
        self._setup_global_hotkey()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # ── Status bar ──────────────────────────────────
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", **pad)
        ttk.Label(status_frame, text="Status:").pack(side="left")
        self.status_var = tk.StringVar(value="Stopped")
        self.status_lbl = ttk.Label(status_frame, textvariable=self.status_var,
                                    foreground="#f38ba8", font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side="left", padx=6)
        ttk.Label(status_frame, text="Toggle hotkey: F8",
                  foreground="#6c7086").pack(side="right")

        # ── Test mode ────────────────────────────────────
        test_frame = ttk.LabelFrame(self.root, text="Test Mode")
        test_frame.pack(fill="x", padx=10, pady=4)

        self.test_mode_var = tk.BooleanVar(value=False)
        test_check = ttk.Checkbutton(
            test_frame, text="Use sample image instead of live screenshot",
            variable=self.test_mode_var, command=self._on_test_mode_toggle
        )
        test_check.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=4)

        ttk.Label(test_frame, text="Image:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.test_path_var = tk.StringVar(value="No image selected")
        ttk.Label(test_frame, textvariable=self.test_path_var,
                  foreground="#6c7086", font=("Segoe UI", 8)).grid(row=1, column=1, sticky="w")

        self.browse_btn = tk.Button(
            test_frame, text="Browse...", command=self._browse_image,
            bg="#313244", fg="#cdd6f4", font=("Segoe UI", 9),
            relief="flat", cursor="hand2", state="disabled"
        )
        self.browse_btn.grid(row=1, column=2, padx=8, pady=4)

        # Auto-load first sample image if folder exists
        self._auto_load_sample()

        # ── Army selection ───────────────────────────────
        army_frame = ttk.LabelFrame(self.root, text="Army")
        army_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(army_frame, text="Select army:").grid(row=0, column=0, padx=8, pady=6, sticky="w")

        self.army_var = tk.StringVar()
        self.army_combo = ttk.Combobox(
            army_frame, textvariable=self.army_var,
            state="readonly", width=20, font=("Segoe UI", 10)
        )
        self.army_combo.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        self.army_combo.bind("<<ComboboxSelected>>", self._on_army_selected)

        refresh_btn = tk.Button(
            army_frame, text="↺", command=self._refresh_armies,
            bg="#313244", fg="#cdd6f4", font=("Segoe UI", 9),
            relief="flat", cursor="hand2"
        )
        refresh_btn.grid(row=0, column=2, padx=4, pady=6)

        self._refresh_armies()

        # ── Resource thresholds ──────────────────────────
        thresh_frame = ttk.LabelFrame(self.root, text="Resource Thresholds")
        thresh_frame.pack(fill="x", padx=10, pady=4)

        # Header row
        ttk.Label(thresh_frame, text="Check", foreground="#89b4fa",
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=8)
        ttk.Label(thresh_frame, text="Resource", foreground="#89b4fa",
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=1, padx=8, sticky="w")
        ttk.Label(thresh_frame, text="Min. Amount", foreground="#89b4fa",
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=8)

        # Resource rows: (label, default, checkbox_var)
        self.resource_checks = {}
        self.thresh_entries  = {}

        resources = [
            ("Gold",        str(_globals["min_gold"]["amount"]),        _globals["min_gold"]["active"]),
            ("Elixir",      str(_globals["min_elixir"]["amount"]),      _globals["min_elixir"]["active"]),
            ("Dark Elixir", str(_globals["min_dark_elixir"]["amount"]), _globals["min_dark_elixir"]["active"]),
        ]

        for row, (label, default, checked) in enumerate(resources, start=1):
            check_var = tk.BooleanVar(value=checked)
            self.resource_checks[label] = check_var

            cb = ttk.Checkbutton(thresh_frame, variable=check_var,
                                 command=self._on_resource_toggle)
            cb.grid(row=row, column=0, padx=8, pady=4)

            ttk.Label(thresh_frame, text=label).grid(row=row, column=1, sticky="w", padx=8, pady=4)

            val_var = tk.StringVar(value=default)
            entry = ttk.Entry(thresh_frame, textvariable=val_var, width=14)
            entry.grid(row=row, column=2, padx=8, pady=4)
            self.thresh_entries[label] = val_var

        apply_btn = tk.Button(
            thresh_frame, text="Apply Thresholds", command=self._apply_thresholds,
            bg="#89b4fa", fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2"
        )
        apply_btn.grid(row=len(resources)+1, column=0, columnspan=3,
                       pady=6, padx=8, sticky="ew")

        # ── Start / Stop buttons ─────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=6)

        self.start_btn = tk.Button(
            btn_frame, text="▶  Start Bot", command=self._start,
            bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2", width=14
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = tk.Button(
            btn_frame, text="■  Stop Bot", command=self._stop,
            bg="#f38ba8", fg="#1e1e2e", font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2", width=14, state="disabled"
        )
        self.stop_btn.pack(side="left")

        # ── Activity log ─────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="Activity Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.log_box = scrolledtext.ScrolledText(
            log_frame, height=14, width=62,
            bg="#181825", fg="#cdd6f4",
            font=("Consolas", 9), state="disabled", relief="flat",
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

        clear_btn = tk.Button(
            log_frame, text="Clear Log", command=self._clear_log,
            bg="#313244", fg="#cdd6f4", font=("Segoe UI", 8),
            relief="flat", cursor="hand2"
        )
        clear_btn.pack(anchor="e", padx=4, pady=(0, 4))

    # ── Army helpers ──────────────────────────
    def _refresh_armies(self):
        """Scan army_strategies/ for army_*.json files and update the dropdown."""
        try:
            files = os.listdir("army_strategies")
            armies = sorted(
                os.path.splitext(f)[0]
                for f in files
                if f.endswith(".json")
                and f.startswith("army_")
                and not f.startswith("ai_")
                and "_reorder" not in f
            )
        except Exception:
            armies = []

        self.army_combo["values"] = armies
        if armies:
            current = self.army_var.get()
            if current not in armies:
                self.army_var.set(armies[0])
                self.bot.update_army(armies[0])
        else:
            self.army_var.set("")
            self._log("[WARN] No army keys found in config.")

    def _on_army_selected(self, _event=None):
        key = self.army_var.get()
        self.bot.update_army(key)
        self._log(f"Army selected: {key}")

    # ── Test mode helpers ─────────────────────
    def _auto_load_sample(self):
        """Auto-select the first image found in image/sample/"""
        if os.path.isdir(SAMPLE_IMAGE_DIR):
            imgs = [f for f in os.listdir(SAMPLE_IMAGE_DIR)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if imgs:
                path = os.path.join(SAMPLE_IMAGE_DIR, imgs[0])
                self.test_path_var.set(os.path.basename(path))
                self.bot.test_image_path = path

    def _on_test_mode_toggle(self):
        enabled = self.test_mode_var.get()
        self.bot.test_mode = enabled
        self.browse_btn.config(state="normal" if enabled else "disabled")
        self._log(f"Test mode {'enabled' if enabled else 'disabled'}.")

    def _browse_image(self):
        path = filedialog.askopenfilename(
            initialdir=SAMPLE_IMAGE_DIR,
            title="Select screenshot",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if path:
            self.bot.test_image_path = path
            self.test_path_var.set(os.path.basename(path))
            self._log(f"Test image set: {os.path.basename(path)}")

    def _on_resource_toggle(self):
        active = [k for k, v in self.resource_checks.items() if v.get()]
        self._log(f"Checking resources: {', '.join(active) if active else 'none'}")

    # ── Hotkey ────────────────────────────────
    def _setup_global_hotkey(self):
        def on_press(key):
            if key == pynput_keyboard.Key.f8:
                self.root.after(0, self._toggle)

        self._listener = pynput_keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    # ── Actions ───────────────────────────────
    def _start(self):
        self._apply_thresholds()
        self.bot.test_mode = self.test_mode_var.get()
        self.bot.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def _stop(self):
        self.bot.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _toggle(self):
        if self.bot.running:
            self._stop()
        else:
            self._start()

    def _apply_thresholds(self):
        try:
            gold  = int(self.thresh_entries["Gold"].get().replace(",", ""))
            elixir = int(self.thresh_entries["Elixir"].get().replace(",", ""))
            dark  = int(self.thresh_entries["Dark Elixir"].get().replace(",", ""))
            self.bot.update_thresholds(
                gold, elixir, dark,
                check_gold        = self.resource_checks["Gold"].get(),
                check_elixir      = self.resource_checks["Elixir"].get(),
                check_dark_elixir = self.resource_checks["Dark Elixir"].get(),
            )
            active = [k for k, v in self.resource_checks.items() if v.get()]
            self._log(f"Thresholds applied. Checking: {', '.join(active) if active else 'none'}")
        except ValueError:
            self._log("[WARN] Invalid threshold value. Please enter numbers only.")

    # ── Helpers ───────────────────────────────
    def _log(self, msg: str):
        def _write():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.root.after(0, _write)

    def _set_status(self, status: str):
        def _update():
            self.status_var.set(status)
            color = "#a6e3a1" if status == "Running" else "#f38ba8"
            self.status_lbl.config(foreground=color)
        self.root.after(0, _update)

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _on_close(self):
        self.bot.stop()
        self._listener.stop()
        self.root.destroy()

