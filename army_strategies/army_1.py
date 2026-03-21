"""
Army 1 deployment strategy.
Standard bleed army: troops around border, spells inward, heroes at left corner.

deploy(bot, diamond, army) is the entry point called by bot_logic.
  bot     – CoCFarmBot instance (provides log, _place_wave, _press_key)
  diamond – [left, top, right, bottom] corner tuples
  army    – dict from _load_army(): {name: {key, count, category}}
"""
import time
import pyautogui

from bot_logic import _perimeter_points, _inward_points, _find_air_defences

SPELLS_PER_AIR_DEF = 3


def deploy(bot, diamond, army):
    troops        = [(n, d) for n, d in army.items() if d["category"] == "troop"]
    wall_breakers = [(n, d) for n, d in army.items() if d["category"] == "wall_breaker"]
    healers       = [(n, d) for n, d in army.items() if d["category"] == "healer"]
    heroes        = [(n, d) for n, d in army.items() if d["category"] == "hero"]
    spells        = [(n, d) for n, d in army.items() if d["category"] == "spell"]

    # 1. First two troop types evenly around border
    for name, data in troops[:2]:
        bot.log(f"[DEPLOY] {name} x{data['count']} around border")
        bot._place_wave(data["key"], _perimeter_points(diamond, data["count"]))

    # 2. Spells: 3 per air defence, remaining inward
    for name, data in spells:
        spell_key   = data["key"]
        total       = data["count"]
        used        = 0

        # Take fresh screenshot and search all air defence templates
        screenshot  = pyautogui.screenshot()
        ad_locs     = _find_air_defences(screenshot)

        bot._press_key(spell_key)   # select spell type once

        if ad_locs:
            bot.log(f"[DEPLOY] {name}: {len(ad_locs)} air defence(s) — placing up to {SPELLS_PER_AIR_DEF} spells each")
            for loc in ad_locs:
                to_place = min(SPELLS_PER_AIR_DEF, total - used)
                if to_place <= 0:
                    break
                for _ in range(to_place):
                    pyautogui.click(*loc)
                    time.sleep(0.2)
                    used += 1
        else:
            bot.log(f"[DEPLOY] {name}: no air defences detected")

        remaining = total - used
        if remaining > 0:
            bot.log(f"[DEPLOY] {name} x{remaining} remaining → inward")
            for x, y in _inward_points(diamond, remaining):
                pyautogui.click(x, y)
                time.sleep(0.1)

    # 4. Heroes — diamond order: [left, top, right, bottom]
    #    AQ + Warden + wall breakers → left corner
    #    other heroes → top, right (skip bottom — troop menu)
    hero_spot = diamond[0]  # left

    aq     = next(((n, d) for n, d in heroes if "archer_queen" in n), None)
    warden = next(((n, d) for n, d in heroes if "grand_warden" in n), None)
    other_heroes = [(n, d) for n, d in heroes
                    if "archer_queen" not in n and "grand_warden" not in n]

    if aq:
        bot.log(f"[DEPLOY] Archer Queen at {hero_spot}")
        bot._press_key(aq[1]["key"])
        pyautogui.click(*hero_spot)
        time.sleep(0.5)

    for _, wb_data in healers:
        bot.log(f"[DEPLOY] Healers x{wb_data['count']} at {hero_spot}")
        bot._press_key(wb_data["key"])
        for _ in range(wb_data["count"]):
            pyautogui.click(*hero_spot)
            time.sleep(0.1)

    if warden:
        bot.log(f"[DEPLOY] Grand Warden at {hero_spot}")
        bot._press_key(warden[1]["key"])
        pyautogui.click(*hero_spot)
        time.sleep(0.5)

    for _, wb_data in wall_breakers:
        bot.log(f"[DEPLOY] Wall Breakers x{wb_data['count']} at {hero_spot}")
        for _ in range(wb_data["count"]):
            bot._press_key(wb_data["key"])
            pyautogui.click(*hero_spot)
            time.sleep(0.15)

    available_corners = [diamond[1], diamond[2]]  # top, right
    for i, (name, data) in enumerate(other_heroes):
        if i >= len(available_corners):
            break
        corner = available_corners[i]
        bot.log(f"[DEPLOY] {name} at corner {corner}")
        bot._press_key(data["key"])
        pyautogui.click(*corner)
        time.sleep(0.5)
