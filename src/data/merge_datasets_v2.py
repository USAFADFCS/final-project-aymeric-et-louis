#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusionne :
- le dataset Roboflow (propre, annoté à la main)
- les pseudo-labels corrigés (auto)
- les hard-negatives (fond sans cocarde)

Pour produire un dataset YOLOv8 final :

data/processed/cocardes_v2/
  images/train, val, test
  labels/train, val, test
"""

import os
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

# === 1. Chemins source ===

# Roboflow export YOLOv8
ROBO_ROOT = ROOT / "data" / "processed" / "cocardes_yolo_manual_15classes"  # adapte si différent
ROBO_TRAIN_IMG = ROBO_ROOT / "train" / "images"
ROBO_TRAIN_LBL = ROBO_ROOT / "train" / "labels"
ROBO_VAL_IMG = ROBO_ROOT / "valid" / "images"  # ou "val"
ROBO_VAL_LBL = ROBO_ROOT / "valid" / "labels"
ROBO_TEST_IMG = ROBO_ROOT / "test" / "images"
ROBO_TEST_LBL = ROBO_ROOT / "test" / "labels"

# Pseudo-labels corrigés
AUTO_IMG = ROOT / "data" / "labeling" / "auto" / "images"
AUTO_LBL = ROOT / "data" / "labeling" / "auto" / "labels"

# Hard negatives (images sans cocarde)
HN_IMG = ROOT / "data" / "processed" / "cocardes_yolo_auto" / "hard_negatives" / "images"

# === 2. Chemins sortie ===

OUT_ROOT = ROOT / "data" / "processed" / "cocardes_merged_v2"
OUT_TRAIN_IMG = OUT_ROOT / "images" / "train"
OUT_TRAIN_LBL = OUT_ROOT / "labels" / "train"
OUT_VAL_IMG = OUT_ROOT / "images" / "val"
OUT_VAL_LBL = OUT_ROOT / "labels" / "val"
OUT_TEST_IMG = OUT_ROOT / "images" / "test"
OUT_TEST_LBL = OUT_ROOT / "labels" / "test"

for d in [OUT_TRAIN_IMG, OUT_TRAIN_LBL, OUT_VAL_IMG, OUT_VAL_LBL, OUT_TEST_IMG, OUT_TEST_LBL]:
    d.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_yolo_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path):
    """Copie un split YOLOv8 (images/labels)."""
    n = 0
    for img_path in src_img.glob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stem = img_path.stem
        lbl_path = src_lbl / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        shutil.copy2(img_path, dst_img / img_path.name)
        shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
        n += 1
    print(f"📦 Copié {n} paires img/label de {src_img} -> {dst_img}")


def copy_auto_to_train():
    """Ajoute les pseudo-labels corrigés au train."""
    n = 0
    for lbl_path in AUTO_LBL.glob("*.txt"):
        stem = lbl_path.stem
        img_path = None
        for ext in IMG_EXTS:
            candidate = AUTO_IMG / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        # pour éviter collisions de noms éventuelles, on préfixe "auto_"
        new_img_name = f"auto_{img_path.name}"
        new_lbl_name = f"auto_{stem}.txt"

        shutil.copy2(img_path, OUT_TRAIN_IMG / new_img_name)
        shutil.copy2(lbl_path, OUT_TRAIN_LBL / new_lbl_name)
        n += 1
    print(f"📦 Ajouté {n} images pseudo-label au train.")


def add_hard_negatives_to_train():
    """Ajoute des images de fond sans cocardes (labels vides)."""
    n = 0
    for img_path in HN_IMG.glob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        new_img_name = f"hn_{img_path.name}"
        new_lbl_name = f"hn_{img_path.stem}.txt"

        shutil.copy2(img_path, OUT_TRAIN_IMG / new_img_name)
        # label vide = pas de box YOLO, mais fichier présent
        (OUT_TRAIN_LBL / new_lbl_name).write_text("", encoding="utf-8")
        n += 1
    print(f"📦 Ajouté {n} hard-negatives au train (labels vides).")


def main():
    print("=== Fusion du dataset cocardes v2 ===")

    # 1) Roboflow -> train/val/test
    copy_yolo_split(ROBO_TRAIN_IMG, ROBO_TRAIN_LBL, OUT_TRAIN_IMG, OUT_TRAIN_LBL)
    copy_yolo_split(ROBO_VAL_IMG, ROBO_VAL_LBL, OUT_VAL_IMG, OUT_VAL_LBL)
    copy_yolo_split(ROBO_TEST_IMG, ROBO_TEST_LBL, OUT_TEST_IMG, OUT_TEST_LBL)

    # 2) Pseudo-labels corrigés -> train uniquement
    copy_auto_to_train()

    # 3) Hard negatives -> train (labels vides)
    add_hard_negatives_to_train()

    print("✅ Dataset fusionné dans", OUT_ROOT)


if __name__ == "__main__":
    main()
