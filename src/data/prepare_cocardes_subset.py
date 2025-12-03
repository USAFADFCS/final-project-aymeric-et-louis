import os
import random
import shutil
from pathlib import Path

# Dossier source : ton dataset avion
CROP_DIR = Path(r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\demos\crop")

# Dossier où on mettra les images à annoter
OUT_DIR = Path(r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\demos\cocardes_to_label")

# Nombre maximum d'images à prendre par type d'avion
MAX_PER_CLASS = 30   # par ex. 30 x ~100 classes ≈ 3000 images

OUT_DIR.mkdir(parents=True, exist_ok=True)

all_images = []

for class_dir in CROP_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    images = [p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not images:
        continue

    random.shuffle(images)
    selected = images[:MAX_PER_CLASS]

    for img_path in selected:
        # On copie avec un nom qui garde la classe pour info (optionnel)
        new_name = f"{class_dir.name}__{img_path.name}"
        dest = OUT_DIR / new_name
        shutil.copy2(img_path, dest)
        all_images.append(dest)

print(f"✅ Copié {len(all_images)} images dans {OUT_DIR}")
