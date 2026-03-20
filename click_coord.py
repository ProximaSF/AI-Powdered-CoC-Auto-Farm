from pynput import mouse

SAVE_FILE = "click_coords.txt"

coords = []

open(SAVE_FILE, "w").close()


def on_click(x, y, _button, pressed):
    if pressed:
        coords.append((x, y))
        print(f"Click #{len(coords)}: ({x}, {y})")
        with open(SAVE_FILE, "w") as f:
            for cx, cy in coords:
                f.write(f"{cx},{cy}\n")


print(f"Listening for clicks. Coords saved to '{SAVE_FILE}'. Press Ctrl+C to stop.")

with mouse.Listener(on_click=on_click) as listener:
    listener.join()
