import os
from pathlib import Path
from shutil import copy2

from ultralytics import YOLO
from tqdm import tqdm

# =========================
# CONFIG À ADAPTER
# =========================

# Modèle YOLO entraîné sur tes cocardes
MODEL_PATH = r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\demos\cocardes\runs\detect\cocardes_v12\weights\best.pt"

# Dossier avec tes ~40 000 images d'avions (structure A10/, F16/, etc.)
SOURCE_DIR = r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\demos\crop"

# Nouveau dataset pseudo-annoté
OUT_DIR = r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\demos\cocardes\auto_dataset"

# Seuil de confiance minimum pour garder une détection
CONF_THRES = 0.7
IOU_THRES = 0.5

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(msg):
    print(msg, flush=True)


def collect_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(dirpath, f))
    return paths


def ensure_out_dirs(base):
    images_dir = os.path.join(base, "auto", "images")
    labels_dir = os.path.join(base, "auto", "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    return images_dir, labels_dir


def main():
    log("\n=== PSEUDO-LABELLING COCARDES (YOLO) ===\n")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    if not os.path.isdir(SOURCE_DIR):
        raise NotADirectoryError(f"Dossier source introuvable : {SOURCE_DIR}")

    log(f"🔧 Chargement du modèle : {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    log(f"📂 Scan des images dans : {SOURCE_DIR}")
    img_paths = collect_images(SOURCE_DIR)
    log(f"   → {len(img_paths)} images trouvées\n")

    images_out, labels_out = ensure_out_dirs(OUT_DIR)

    kept = 0
    skipped = 0

    for img_path in tqdm(img_paths, desc="Pseudo-labelling"):
        # Inference sur UNE image (évite le problème 'too many open files')
        results = model.predict(
            source=img_path,
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=640,
            device=0,        # GPU 0
            stream=False,
            verbose=False
        )

        r0 = results[0]
        boxes = r0.boxes

        # Si aucune détection → on ignore cette image pour ce dataset
        if boxes is None or len(boxes) == 0:
            skipped += 1
            continue

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xywhn = boxes.xywhn.cpu().numpy()  # [N, 4] normalisé [0,1]

        # Construire les lignes YOLO
        lines = []
        for cid, (xc, yc, w, h) in zip(cls_ids, xywhn):
            lines.append(f"{int(cid)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if not lines:
            skipped += 1
            continue

        # Nom de fichier de sortie (on "aplatit" la structure)
        img_name = Path(img_path).name
        stem = Path(img_name).stem

        img_dst = os.path.join(images_out, img_name)
        lbl_dst = os.path.join(labels_out, stem + ".txt")

        # Copier l'image dans le nouveau dataset
        copy2(img_path, img_dst)

        # Écrire le label YOLO
        with open(lbl_dst, "w", encoding="utf-8") as f:
            f.writelines(lines)

        kept += 1

        if kept % 500 == 0:
            log(f"   → {kept} images pseudo-annotées, {skipped} ignorées")

    log("\n✅ Pseudo-labellisation terminée.")
    log(f"   Images gardées (>= 1 cocarde détectée): {kept}")
    log(f"   Images ignorées : {skipped}")
    log(f"   Nouveau dataset : {OUT_DIR}\n")


if __name__ == "__main__":
    main()
