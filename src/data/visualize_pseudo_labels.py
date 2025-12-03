import os
import cv2
from tqdm import tqdm

# --- PARAMÈTRES À MODIFIER SI BESOIN ---
DATASET_PATH = r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\demos\cocardes\auto_dataset\train"
OUT_PATH = os.path.join(DATASET_PATH, "../../data/labeling/auto/viz")
os.makedirs(OUT_PATH, exist_ok=True)

# Chargement du mapping des classes
# Optionnel : remplacer par ton dictionnaire exact
CLASS_NAMES = [
    "australia", "china", "france", "india", "japan",
    "russia", "south_korea", "turkey", "uk", "ukraine", "usa"
]

IMG_DIR = os.path.join(DATASET_PATH, "../../data/labeling/auto/images")
LBL_DIR = os.path.join(DATASET_PATH, "../../data/labeling/auto/labels")

def main():
    images = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
    print(f"🖼️ {len(images)} images trouvées")
    print(f"💾 Les images annotées seront enregistrées dans : {OUT_PATH}\n")

    for img_name in tqdm(images, desc="Visualisation"):
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

        # Charger l’image
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Lire les labels
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                cls, x, y, bw, bh = map(float, line.split())

                cls = int(cls)
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                # Dessiner la box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

                label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls_{cls}"
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Sauver l’image annotée
        out_path = os.path.join(OUT_PATH, img_name)
        cv2.imwrite(out_path, img)

    print("\n✅ Visualisation terminée.")
    print(f"➡️ Résultats disponibles dans : {OUT_PATH}")

if __name__ == "__main__":
    main()
