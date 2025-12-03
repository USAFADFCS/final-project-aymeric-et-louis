#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import gradio as gr

# ==========================
# CONFIG
# ==========================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_ROOT = ROOT_DIR / "data/labeling/auto"
DATASET_SAVE_ROOT = ROOT_DIR / "data/processed/cocardes_yolo_auto"

IMG_DIR = DATASET_ROOT / "images"
LBL_DIR = DATASET_ROOT / "labels"

HN_IMG_DIR = DATASET_SAVE_ROOT / "hard_negatives" / "images"
HN_IMG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "australia",
    "china",
    "france",
    "india",
    "japan",
    "russia",
    "south_korea",
    "turkey",
    "uk",
    "ukraine",
    "usa",
    "germany",
    "italy",
    "egypt",
    "israel",
]

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

# ==========================
# LOGIQUE
# ==========================

def list_items() -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []

    for lbl_name in os.listdir(LBL_DIR):
        if not lbl_name.endswith(".txt"):
            continue
        stem = Path(lbl_name).stem
        lbl_path = str(LBL_DIR / lbl_name)

        img_path = None
        for ext in IMG_EXTS:
            candidate = IMG_DIR / f"{stem}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path:
            items.append((img_path, lbl_path))

    items.sort()
    print(f"🔎 {len(items)} images trouvées pour la review.")
    return items


ITEMS: List[Tuple[str, str]] = list_items()
N_ITEMS = len(ITEMS)


def summarize_classes(lbl_path: str) -> str:
    """Affiche les classes détectées en gros."""
    if not os.path.exists(lbl_path):
        return "**Pas de label**"

    counts: Dict[int, int] = {}

    with open(lbl_path, "r", encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            counts[cid] = counts.get(cid, 0) + 1

    if not counts:
        return "**Aucune box détectée**"

    chunks = []
    for cid, n in counts.items():
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"id={cid}"
        chunks.append(f"**{name}** ({n})")

    return "Classes détectées : " + ", ".join(chunks)


def draw_boxes(img_path: str, lbl_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Impossible de lire l’image :", img_path)
        return None

    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        lines = open(lbl_path, "r", encoding="utf-8").readlines()
    else:
        lines = []

    for l in lines:
        parts = l.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        x, y, bw, bh = map(float, parts[1:])
        cx, cy = x * w, y * h
        ww, hh = bw * w, bh * h
        x1, y1 = int(cx - ww / 2), int(cy - hh / 2)
        x2, y2 = int(cx + ww / 2), int(cy + hh / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"id={cls_id}"

        cv2.putText(img, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_item(idx: int):
    if N_ITEMS == 0:
        return None, "0/0 — Fin", "", 0

    if idx < 0:
        idx = 0
    if idx >= N_ITEMS:
        return None, f"Fin ({N_ITEMS}/{N_ITEMS})", "", idx

    img_path, lbl_path = ITEMS[idx]
    img = draw_boxes(img_path, lbl_path)
    info = f"{idx+1}/{N_ITEMS} — {Path(img_path).name}"
    classes_text = summarize_classes(lbl_path)

    return img, info, classes_text, idx


def action_keep(idx: int):
    return load_item(idx + 1)


def action_correct(idx: int, new_class_name: str):
    if idx < 0 or idx >= N_ITEMS:
        return load_item(idx)

    _, lbl_path = ITEMS[idx]
    cls_id = CLASS_NAMES.index(new_class_name)

    new_lines = []
    for l in open(lbl_path, "r", encoding="utf-8"):
        p = l.strip().split()
        if len(p) < 5:
            continue
        p[0] = str(cls_id)
        new_lines.append(" ".join(p) + "\n")

    open(lbl_path, "w", encoding="utf-8").writelines(new_lines)

    return load_item(idx + 1)


def action_noise(idx: int):
    global ITEMS, N_ITEMS

    if idx < 0 or idx >= N_ITEMS:
        return load_item(idx)

    img_path, lbl_path = ITEMS[idx]
    os.replace(img_path, HN_IMG_DIR / Path(img_path).name)

    if os.path.exists(lbl_path):
        os.remove(lbl_path)

    del ITEMS[idx]
    N_ITEMS = len(ITEMS)

    return load_item(idx)


def action_jump(go_to: int):
    idx = max(0, min(N_ITEMS - 1, go_to - 1))
    return load_item(idx)


# ==========================
# UI
# ==========================

def build_ui():
    with gr.Blocks(title="Revue pseudo-labels cocardes") as demo:
        gr.Markdown("### Revue des pseudo-labels")

        state_idx = gr.State(0)

        with gr.Row():
            img_out = gr.Image(type="numpy")
            with gr.Column():
                info = gr.Markdown()
                classes_md = gr.Markdown()

                jump_num = gr.Number(label="Aller à l’image n°")
                jump_btn = gr.Button("Aller")

                cls_dropdown = gr.Dropdown(CLASS_NAMES, value="france")
                keep_btn = gr.Button("Valider tel quel")
                correct_btn = gr.Button("Corriger")
                noise_btn = gr.Button("Hard negative")

        demo.load(fn=lambda: load_item(0),
                  inputs=None,
                  outputs=[img_out, info, classes_md, state_idx])

        keep_btn.click(action_keep, inputs=state_idx,
                       outputs=[img_out, info, classes_md, state_idx])

        correct_btn.click(action_correct,
                          inputs=[state_idx, cls_dropdown],
                          outputs=[img_out, info, classes_md, state_idx])

        noise_btn.click(action_noise, inputs=state_idx,
                        outputs=[img_out, info, classes_md, state_idx])

        jump_btn.click(action_jump, inputs=jump_num,
                       outputs=[img_out, info, classes_md, state_idx])

    return demo


if __name__ == "__main__":
    print("Dataset :", DATASET_ROOT)
    print("Images :", N_ITEMS)
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
