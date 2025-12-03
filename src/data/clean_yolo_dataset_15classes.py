import os
import shutil
from pathlib import Path

import yaml

# ==== CONFIG ====
ROOT_DIR = Path(__file__).resolve().parents[2]

# Dossier d'origine (Roboflow) — NON MODIFIÉ
SRC_DATASET_DIR = ROOT_DIR / "data" / "raw" / "roboflow_cocardes"

# Dossier de sortie (copie traitée)
DST_DATASET_DIR = ROOT_DIR / "data" / "processed" / "cocardes_yolo_manual_15classes"
# =================

# Ordre CIBLE des classes (indices 0..14) tel que tu le veux
TARGET_CLASS_ORDER = [
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
    "italia",   # ⚠️ nom dans le dataset d'origine
    "egypt",
    "israel",
]

# Noms à écrire dans le YAML final (on corrige italia -> italy ici)
FINAL_YOLO_NAMES = [
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
    "italy",   # nom corrigé pour le YAML final
    "egypt",
    "israel",
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def clean_split(split_path, old_to_new_class):
    labels_path = os.path.join(split_path, "labels")
    images_path = os.path.join(split_path, "images")

    if not os.path.isdir(labels_path):
        print(f"[WARN] Pas de dossier labels pour {split_path}, on passe.")
        return 0, 0, 0

    removed = 0
    empty = 0
    kept = 0

    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_path, label_file)
        image_base = label_file.replace(".txt", "")
        image_path_jpg = os.path.join(images_path, image_base + ".jpg")
        image_path_png = os.path.join(images_path, image_base + ".png")

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Cas label vide → suppression label + image associée
        if len(lines) == 0:
            empty += 1
            os.remove(label_path)
            if os.path.exists(image_path_jpg):
                os.remove(image_path_jpg)
            if os.path.exists(image_path_png):
                os.remove(image_path_png)
            continue

        new_lines = []
        valid = True

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            cls_id = int(parts[0])

            # Si la classe n'est pas dans le mapping, on jette TOUT le fichier
            if cls_id not in old_to_new_class:
                valid = False
                break

            # Réécriture avec la nouvelle classe réindexée (0..14)
            parts[0] = str(old_to_new_class[cls_id])
            new_lines.append(" ".join(parts))

        # Si aucune bbox conservée ou classe non désirée → supprimer le fichier + image
        if not valid or len(new_lines) == 0:
            removed += 1
            os.remove(label_path)
            if os.path.exists(image_path_jpg):
                os.remove(image_path_jpg)
            if os.path.exists(image_path_png):
                os.remove(image_path_png)
            continue

        # Écriture du label filtré
        with open(label_path, "w") as f:
            f.write("\n".join(new_lines))
        kept += 1

    return kept, removed, empty


def main():
    # 1) Copie du dataset d'origine vers le dossier de sortie
    if os.path.exists(DST_DATASET_DIR):
        raise RuntimeError(f"Le dossier de sortie existe déjà : {DST_DATASET_DIR}\n"
                           f"Supprime-le ou change DST_DATASET_DIR pour éviter d'écraser des données.")
    print(f"📂 Copie du dataset de\n  {SRC_DATASET_DIR}\nvers\n  {DST_DATASET_DIR}")
    shutil.copytree(SRC_DATASET_DIR, DST_DATASET_DIR)

    # 2) Travail uniquement sur la copie
    data_yaml_path = os.path.join(DST_DATASET_DIR, "data.yaml")
    data = load_yaml(data_yaml_path)

    all_classes = data["names"]  # liste des 35 classes d'origine
    print("Classes d'origine (35) :", all_classes)

    # Vérif qu'on a bien les classes cible dans le dataset d'origine
    name_to_old_id = {name: i for i, name in enumerate(all_classes)}

    missing = [c for c in TARGET_CLASS_ORDER if c not in name_to_old_id]
    if missing:
        raise ValueError(f"Les classes suivantes n'existent pas dans le YAML d'origine : {missing}")

    # Construction du mapping ancien_index -> nouvel_index (0..14)
    old_to_new_class = {}
    for new_id, class_name in enumerate(TARGET_CLASS_ORDER):
        old_id = name_to_old_id[class_name]
        old_to_new_class[old_id] = new_id

    print("Mapping ancien_id -> nouveau_id :", old_to_new_class)

    # 3) Nettoyage des splits sur la COPIE
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(DST_DATASET_DIR, split)
        if not os.path.exists(split_path):
            continue

        print(f"\n=== Traitement du split: {split.upper()} ===")
        kept, removed, empty = clean_split(split_path, old_to_new_class)
        print("Labels gardés   :", kept)
        print("Labels supprimés :", removed)
        print("Labels vides     :", empty)

    # 4) Mise à jour du YAML dans la copie
    data["nc"] = len(FINAL_YOLO_NAMES)
    data["names"] = FINAL_YOLO_NAMES
    save_yaml(data_yaml_path, data)

    print("\n🎉 Terminé !")
    print(f"- Dataset d'origine INTact : {SRC_DATASET_DIR}")
    print(f"- Dataset traité (15 classes) : {DST_DATASET_DIR}")


if __name__ == "__main__":
    main()
