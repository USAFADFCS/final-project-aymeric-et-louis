import os
import yaml

# ==== CONFIG ====
DATASET_DIR = r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\CocardesV2.v2i.yolov8"     # <-- mets ici ton dossier YOLO (celui qui contient auto/, valid/, test/)
CLASSES_TO_KEEP = [
    "usa", "russia", "china", "japan", "france", "uk",
    "turkey", "australia", "south_korea", "ukraine", "india"
]
# ===============

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def clean_split(split_path, old_to_new_class):
    labels_path = os.path.join(split_path, "labels")
    images_path = os.path.join(split_path, "images")

    removed = 0
    empty = 0
    kept = 0

    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_path, label_file)
        image_path = os.path.join(images_path, label_file.replace(".txt", ".jpg"))

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Cas label vide → suppression
        if len(lines) == 0:
            empty += 1
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            continue

        new_lines = []
        valid = True

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])

            if cls_id not in old_to_new_class:
                valid = False
                break

            # Réécriture avec la nouvelle classe réindexée
            parts[0] = str(old_to_new_class[cls_id])
            new_lines.append(" ".join(parts))

        # Si aucune bbox conservée → supprimer le fichier
        if not valid or len(new_lines) == 0:
            removed += 1
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            continue

        # Écriture du label filtré
        with open(label_path, "w") as f:
            f.write("\n".join(new_lines))
        kept += 1

    return kept, removed, empty


def main():
    data_yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    data = load_yaml(data_yaml_path)

    all_classes = data["names"]

    # Liste des classes à garder → indices YOLO d'origine
    old_class_ids_to_keep = [i for i, name in enumerate(all_classes) if name in CLASSES_TO_KEEP]

    print("Classes disponibles :", all_classes)
    print("Classes conservées :", CLASSES_TO_KEEP)
    print("Indices conservés  :", old_class_ids_to_keep)

    # Mapping ancien_index → nouveau_index
    old_to_new_class = {old: new for new, old in enumerate(old_class_ids_to_keep)}

    print("Mapping :", old_to_new_class)

    # Nettoyage des splits
    for split in ["auto", "valid", "test"]:
        split_path = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_path):
            continue

        kept, removed, empty = clean_split(split_path, old_to_new_class)
        print(f"\n=== {split.upper()} ===")
        print("Labels gardés   :", kept)
        print("Labels supprimés :", removed)
        print("Labels vides     :", empty)

    # Mise à jour du YAML
    data["names"] = [all_classes[i] for i in old_class_ids_to_keep]
    save_yaml(data_yaml_path, data)

    print("\n🎉 Nettoyage terminé ! Dataset prêt pour l'entraînement YOLO.")


if __name__ == "__main__":
    main()
