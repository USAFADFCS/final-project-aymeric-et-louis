import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import pipeline

# Optionnel: pour CSV/affichage
try:
    import pandas as pd
except ImportError:
    pd = None


def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Classification d'avions militaires (Hugging Face).")
    parser.add_argument("--input", required=True, help="Dossier d'images ou image unique")
    parser.add_argument("--model", default="Illia56/Illia56-Military-Aircraft-Detection",
                        help="Identifiant du modèle HF (classification)")
    parser.add_argument("--topk", type=int, default=5, help="Top-k classes à afficher")
    parser.add_argument("--csv", default="results.csv", help="Chemin du CSV de sortie (désactiver avec --csv '')")
    parser.add_argument("--show", action="store_true", help="Afficher l'image et la prédiction top-1")
    args = parser.parse_args()

    # Pipeline HF (télécharge auto les poids)
    clf = pipeline("image-classification", model=args.model)

    in_path = Path(args.input)
    if in_path.is_dir():
        images = [p for p in sorted(in_path.iterdir()) if p.is_file() and is_image(p)]
    elif in_path.is_file() and is_image(in_path):
        images = [in_path]
    else:
        raise SystemExit("Chemin d'entrée invalide. Fournis un dossier ou une image (.jpg/.png/...).")

    rows = []
    for img_path in tqdm(images, desc="Inférence"):
        # On peut passer un chemin directement au pipeline
        preds = clf(str(img_path), top_k=args.topk)

        # preds = [{'label': 'F-16', 'score': 0.93}, ...]
        top1 = preds[0]["label"] if preds else "N/A"
        print(f"{img_path.name} -> Top1: {top1} | "
              + ", ".join([f"{p['label']}:{p['score']:.2f}" for p in preds]))

        # Pour CSV
        row = {"file": img_path.name}
        for i, p in enumerate(preds, 1):
            row[f"top{i}_label"] = p["label"]
            row[f"top{i}_score"] = round(float(p["score"]), 4)
        rows.append(row)

        if args.show:
            try:
                import matplotlib.pyplot as plt
                im = Image.open(img_path).convert("RGB")
                plt.figure()
                plt.imshow(im)
                plt.axis("off")
                plt.title(f"{img_path.name} — Top1: {top1}")
                plt.show()
            except ImportError:
                pass

    # Sauvegarde CSV si demandé et pandas dispo
    if args.csv and pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv("outputs\\" + args.csv, index=False, encoding="utf-8")
        print(f"\nCSV écrit: {args.csv}")


if __name__ == "__main__":
    main()