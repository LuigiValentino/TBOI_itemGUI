import cv2
import os
import pyautogui
import numpy as np
import tkinter as tk
from tkinter import Canvas, Frame
from PIL import Image, ImageTk
import requests
from bs4 import BeautifulSoup
import threading

item_folder = "./item_sprites"
orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, edgeThreshold=15)

paused = False

def load_items(folder_path):
    items = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            item_name = file_name[:-4]
            img = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                kp, des = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY), None)
                items[item_name] = {'image': img, 'keypoints': kp, 'descriptors': des}
    return items

def detect_item_in_region(items, region):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    screenshot = cv2.resize(screenshot, (128, 128), interpolation=cv2.INTER_AREA)
    kp2, des2 = orb.detectAndCompute(screenshot, None)

    if des2 is None or len(des2) < 2:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best_match = None
    best_score = float('inf')

    for item_name, item_data in items.items():
        des1 = item_data['descriptors']
        if des1 is None or len(des1) < 2:
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            avg_distance = np.mean([m.distance for m in good_matches])
            if avg_distance < best_score:
                best_score = avg_distance
                best_match = item_name

    if best_match:
        return best_match, best_score
    else:
        return None, None

item_info_cache = {}

def get_item_info(item_name):
    if item_name in item_info_cache:
        return item_info_cache[item_name]

    base_url = 'https://bindingofisaacrebirth.fandom.com/wiki/'
    item_url = base_url + item_name.replace(' ', '_')
    response = requests.get(item_url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    title_element = soup.find('h1', {'class': 'page-header__title'})
    title = title_element.text.strip() if title_element else item_name

    quality_element = soup.find('div', {'data-source': 'quality'})
    quality_value = quality_element.find('div', {'class': 'pi-data-value'}).text.strip() if quality_element else '0'

    entity_id_element = soup.find('div', {'data-source': 'id'})
    entity_id = entity_id_element.find('div', {'class': 'pi-data-value'}).text.strip() if entity_id_element else "Unknown"

    quote_element = soup.find('div', {'data-source': 'quote'})
    pickup_quote = quote_element.find('div', {'class': 'pi-data-value'}).text.strip() if quote_element else "No Pickup Quote"

    effects_list = []
    effect_header = soup.find('span', {'id': 'Effects'})
    if effect_header:
        effects_ul = effect_header.find_next('ul')
        if effects_ul:
            effects_list = [li.text.strip() for li in effects_ul.find_all('li')]

    synergies_list = []
    synergy_header = soup.find('span', {'id': 'Synergies'})
    if synergy_header:
        synergies_ul = synergy_header.find_next('ul')
        if synergies_ul:
            synergies_list = [li.text.strip() for li in synergies_ul.find_all('li')]

    try:
        quality = int(quality_value)
        if quality < 0 or quality > 4:
            quality = 0
    except ValueError:
        quality = 0

    item_info = {
        'title': title,
        'quality': quality,
        'entity_id': entity_id,
        'pickup_quote': pickup_quote,
        'effects': effects_list,
        'synergies': synergies_list
    }
    item_info_cache[item_name] = item_info
    return item_info

def update_stars(tier):
    try:
        tier = int(tier)
    except ValueError:
        tier = 0

    tier = 4 - tier

    for i in range(4):
        if i < tier:
            star_image = ImageTk.PhotoImage(star_empty_img)
        else:
            star_image = ImageTk.PhotoImage(star_filled_img)
        stars[i].config(image=star_image)
        stars[i].image = star_image

def fetch_and_update():
    item_info = get_item_info(detected_item)
    if item_info:
        title = item_info.get('title', detected_item)
        quality = item_info.get('quality', 0)
        entity_id = item_info.get('entity_id', 'Unknown')
        pickup_quote = item_info.get('pickup_quote', 'No Pickup Quote')
        effects = item_info.get('effects', [])
        synergies = item_info.get('synergies', [])

        detected_label.config(
            text=title if title else "No Title",
            width=30,
            anchor="center"
        )
        entity_id_label.config(
            text=f"Entity ID: {entity_id if entity_id else 'Unknown'}",
            width=30,
            anchor="center"
        )
        pickup_quote_label.config(
            text=pickup_quote if pickup_quote else "No Pickup Quote",
            wraplength=350,
            width=30,
            anchor="center"
        )

        effects_text = "\n".join(effects) if effects else "No Effects"
        effects_label.config(
            text=effects_text,
            wraplength=350,
            justify="left",
            anchor="w",
            padx=10
        )

        synergies_text = "\n".join(synergies) if synergies else "No Synergies"
        synergies_label.config(
            text=synergies_text,
            wraplength=350,
            justify="left",
            anchor="w",
            padx=10
        )

        update_stars(quality)
    else:
        detected_label.config(text="No item detected", width=30, anchor="center")
        entity_id_label.config(text="Entity ID: Unknown", width=30, anchor="center")
        pickup_quote_label.config(text="No Pickup Quote", wraplength=350, width=30, anchor="center")
        effects_label.config(text="No Effects", wraplength=350, justify="left", anchor="w", padx=10)
        synergies_label.config(text="No Synergies", wraplength=350, justify="left", anchor="w", padx=10)
        update_stars(0)

def update_interface():
    if paused:
        root.after(200, update_interface)
        return
    x, y = pyautogui.position()
    region_size = 50
    left = max(x - region_size // 2, 0)
    top = max(y - region_size // 2, 0)
    region = (left, top, region_size, region_size)
    global detected_item
    detected_item, score = detect_item_in_region(items, region)

    if detected_item:
        item_data = items[detected_item]
        item_image = item_data['image']
        item_image = cv2.resize(item_image, (128, 128), interpolation=cv2.INTER_AREA)
        item_image_pil = Image.fromarray(item_image)
        item_image_tk = ImageTk.PhotoImage(item_image_pil)
        item_label.config(image=item_image_tk)
        item_label.image = item_image_tk

        threading.Thread(target=fetch_and_update).start()
    else:
        detected_label.config(text="No item detected", width=30, anchor="center")
        entity_id_label.config(text="Entity ID: Unknown", width=30, anchor="center")
        pickup_quote_label.config(text="No Pickup Quote", width=30, anchor="center")
        effects_label.config(text="No Effects", wraplength=350, justify="left", anchor="w", padx=10)
        synergies_label.config(text="No Synergies", wraplength=350, justify="left", anchor="w", padx=10)
        item_label.config(image="")
        update_stars(0)

    root.after(200, update_interface)

def toggle_pause(event=None):
    global paused
    paused = not paused
    if paused:
        root.config(bg="white", highlightbackground="red", highlightthickness=2)
    else:
        root.config(bg="white", highlightbackground="white", highlightthickness=0)

def adjust_wraplength(event):
    new_width = scroll_canvas.winfo_width() - 20
    effects_label.config(wraplength=new_width)
    synergies_label.config(wraplength=new_width)

items = load_items(item_folder)

star_filled_img = Image.open("GUI_sources/estrella_vacia.webp")
star_empty_img = Image.open("./GUI_sources/estrella.webp")

root = tk.Tk()
root.title("TBOI Item GUI")
root.geometry("400x600")
root.resizable(False, False)
root.bind("<space>", toggle_pause)
root.bind('<Configure>', adjust_wraplength)

scroll_canvas = Canvas(root, width=400, height=600)
scroll_frame = Frame(scroll_canvas)
scrollbar = tk.Scrollbar(root, orient="vertical", command=scroll_canvas.yview)
scroll_canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
scroll_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))

item_label = tk.Label(scroll_frame)
item_label.pack(pady=10)

stars_frame = tk.Frame(scroll_frame)
stars_frame.pack(pady=5)
stars = []
for _ in range(4):
    star_label = tk.Label(stars_frame)
    star_label.pack(side=tk.LEFT, padx=2)
    stars.append(star_label)

separator1 = tk.Frame(scroll_frame, height=2, bd=1, relief=tk.SUNKEN)
separator1.pack(fill=tk.X, padx=5, pady=5)

detected_label = tk.Label(scroll_frame, text="Waiting for detection...", font=("Arial", 14), width=30, anchor="center")
detected_label.pack()

separator2 = tk.Frame(scroll_frame, height=2, bd=1, relief=tk.SUNKEN)
separator2.pack(fill=tk.X, padx=5, pady=5)

entity_id_label = tk.Label(scroll_frame, text="Entity ID: Unknown", font=("Arial", 12), width=30, anchor="center")
entity_id_label.pack()

separator3 = tk.Frame(scroll_frame, height=2, bd=1, relief=tk.SUNKEN)
separator3.pack(fill=tk.X, padx=5, pady=5)

pickup_quote_label = tk.Label(scroll_frame, text="No Pickup Quote", font=("Arial", 12, "italic"), width=30, anchor="center")
pickup_quote_label.pack()

separator4 = tk.Frame(scroll_frame, height=2, bd=1, relief=tk.SUNKEN)
separator4.pack(fill=tk.X, padx=5, pady=5)

effects_title_label = tk.Label(scroll_frame, text="Effects", font=("Arial", 12, "bold"), anchor="center")
effects_title_label.pack()

effects_label = tk.Label(
    scroll_frame,
    text="No Effects",
    font=("Arial", 12),
    justify="left",
    wraplength=350,
    anchor="w",
    padx=10
)
effects_label.pack(pady=5)

separator5 = tk.Frame(scroll_frame, height=2, bd=1, relief=tk.SUNKEN)
separator5.pack(fill=tk.X, padx=5, pady=5)

synergies_title_label = tk.Label(scroll_frame, text="Synergies", font=("Arial", 12, "bold"), anchor="center")
synergies_title_label.pack()

synergies_label = tk.Label(
    scroll_frame,
    text="No Synergies",
    font=("Arial", 12),
    justify="left",
    wraplength=350,
    anchor="w",
    padx=10
)
synergies_label.pack(pady=5)

icon_path = "./GUI_sources/icon.ico"
if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

root.after(100, update_interface)
root.mainloop()
