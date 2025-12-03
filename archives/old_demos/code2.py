#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from PIL import Image

# YOLO (ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

# =========================
# Config utilisateur
# =========================
# Chemins (à adapter)
MODEL_YOLO_PATH = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis-1/demos/yolov8n.pt"            # si vous avez un .pt ultralytics, mettez le .pt (ex: "yolov8n.pt")
MODEL_RESNET_PATH = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis-1/demos/resnet_model_optimized.pth"
IMAGE_PATH = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis-1/demos/inputs/a400m.jpeg"

# Optionnel: liste des classes (si vous avez un fichier JSON/texte)
# Si None, on essaiera de la tirer du checkpoint (clé 'classes' si présente)
CLASSES: Optional[List[str]] = None  # par exemple: ["A320", "A330", ...] (taille 95)

# Seuils
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45

# Tailles d'entrée ResNet
RESNET_INPUT_SIZE = 224

data_dir = "/Users/a.cariven/Documents/USAFA/comp scien/crop"

CLASSES = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])

# =========================
# Gestion des devices
# =========================
def get_device_labels():
    device_yolo = "cpu"  # imposé par la demande
    device_resnet = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return device_yolo, device_resnet

# =========================
# Utils
# =========================
def log(s: str):
    print(s, flush=True)

def timeit(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = fn(*args, **kwargs)
        dt = (time.time() - t0) * 1000
        log(f"⏱️  {fn.__name__} terminé en {dt:.1f} ms")
        return res
    return wrapper

def ensure_exists(path: str, kind: str = "fichier"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le {kind} '{path}' est introuvable.")
    
# --- à ajouter près des autres utils ---
def no_airplane_result():
    return {
        "contains_airplane": False,
        "message": "Aucun avion détecté",
        "results": []
    }


# =========================
# Pré-traitements ResNet
# =========================
def build_resnet_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

# =========================
# Chargeur ResNet auto-détectant
# =========================
def _strip_prefix(state_dict, prefixes=("module.", "models.")):
    new_sd = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        new_sd[k] = v
    return new_sd

def _is_bottleneck(state_keys):
    # Bottleneck ⇒ présence de conv3 dans les blocs
    return any("layer1.0.conv3.weight" in k for k in state_keys)

def _infer_layout(state_keys):
    # Estime le nombre de blocs par layer via les index présents
    def count_blocks(layer):
        pattern = re.compile(rf"^{layer}\.(\d+)\.")
        idx = set()
        for k in state_keys:
            m = pattern.match(k)
            if m:
                idx.add(int(m.group(1)))
        return (1 + max(idx)) if idx else 0
    return tuple(map(count_blocks, ["layer1", "layer2", "layer3", "layer4"]))

def _build_matching_resnet(state_dict, num_classes_hint=None):
    state_dict = _strip_prefix(state_dict)
    keys = list(state_dict.keys())

    is_bottleneck = _is_bottleneck(keys)
    layout = _infer_layout(keys)  # e.g., (2,2,2,2) / (3,4,6,3) / (3,8,36,3)

    # Nb classes depuis fc.weight si possible
    num_classes_ckpt = None
    if "fc.weight" in state_dict and isinstance(state_dict["fc.weight"], torch.Tensor):
        num_classes_ckpt = state_dict["fc.weight"].shape[0]
    num_classes = num_classes_hint or num_classes_ckpt or 1000

    # Choix modèle
    if is_bottleneck:
        # 50/101/152 (fc.in_features=2048)
        if layout == (3,4,6,3):
            model = models.resnet50(weights=None)
        elif layout == (3,4,23,3):
            model = models.resnet101(weights=None)
        elif layout == (3,8,36,3):
            model = models.resnet152(weights=None)
        else:
            model = models.resnet50(weights=None)
        in_feats = model.fc.in_features
    else:
        # 18/34 (fc.in_features=512)
        if layout == (2,2,2,2):
            model = models.resnet18(weights=None)
        elif layout == (3,4,6,3):
            model = models.resnet34(weights=None)
        else:
            model = models.resnet34(weights=None)
        in_feats = model.fc.in_features

    # Adapter la couche finale pour charger correctement fc
    model.fc = nn.Linear(in_feats, num_classes)
    return model, state_dict, num_classes

def _extract_state_dict(ckpt_obj):
    # Supporte formats: state_dict direct, ou dict avec 'state_dict'
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"], ckpt_obj.get("classes", None)
        # certains exports utilisent 'models' ou 'net'
        for k in ("models", "net"):
            if k in ckpt_obj and hasattr(ckpt_obj[k], "state_dict"):
                return ckpt_obj[k].state_dict(), ckpt_obj.get("classes", None)
    return ckpt_obj, None

@timeit
def load_resnet_autodetect(model_path: str, classes: Optional[List[str]], device: str):
    ensure_exists(model_path, "checkpoint ResNet")
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict, classes_in_ckpt = _extract_state_dict(ckpt)

    # classes hint
    num_classes_hint = len(classes) if isinstance(classes, list) else None
    model, state_dict, num_classes = _build_matching_resnet(state_dict, num_classes_hint)

    if classes_in_ckpt is not None and isinstance(classes_in_ckpt, (list, tuple)):
        if classes is None:
            classes = list(classes_in_ckpt)
        elif len(classes) != len(classes_in_ckpt):
            log(f"⚠️ Liste de classes fournie ({len(classes)}) ≠ checkpoint ({len(classes_in_ckpt)}). "
                f"On conserve la dimension du checkpoint pour charger fc.")

    log(f"ℹ️ ResNet détecté: "
        f"{'Bottleneck' if model.layer1[0].expansion==4 else 'BasicBlock'} | "
        f"fc.in_features={model.fc.in_features} | classes={num_classes}")

    # Chargement strict
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model, classes

# =========================
# Chargement YOLO (CPU)
# =========================
@timeit
def load_yolo(model_path: str, device: str = "cpu"):
    if YOLO is None:
        raise RuntimeError("ultralytics n'est pas installé. Installez-le avec: pip install ultralytics")
    ensure_exists(model_path, "modèle YOLO")
    # ultralytics supporte .pt/.onnx. Le 'device' est géré à l'inférence.
    model = YOLO(model_path)
    return model

# =========================
# Inference helpers
# =========================
def yolo_detect(yolo_model, image_path: str, conf: float, iou: float) -> List[Dict[str, Any]]:
    results = yolo_model.predict(
        source=image_path,
        device="cpu",
        conf=conf,
        iou=iou,
        verbose=False
    )
    dets = []
    if not results:
        return dets
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return dets
    boxes = r0.boxes.xyxy.cpu().numpy()
    scores = r0.boxes.conf.cpu().numpy()
    cls_ids = r0.boxes.cls.cpu().numpy() if r0.boxes.cls is not None else None
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        dets.append({
            "bbox": (x1, y1, x2, y2),
            "score": float(scores[i]),
            "cls_id": int(cls_ids[i]) if cls_ids is not None else -1
        })
    return dets

def crop_image(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
    return pil_img.crop((x1, y1, x2, y2))

def classify_topk(model_resnet,
                  device: str,
                  crops: List[Image.Image],
                  classes: Optional[List[str]],
                  img_size: int = 224,
                  k: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Retourne pour chaque crop une liste (triée desc) des k meilleures classes:
    [
      [ {"label": str, "prob": float, "class_id": int}, ... (k) ],
      ...
    ]
    """
    if len(crops) == 0:
        return []

    tfm = build_resnet_transform(img_size)
    with torch.no_grad():
        batch = torch.stack([tfm(im) for im in crops]).to(device)  # [N,3,H,W]
        logits = model_resnet(batch)                                # [N,C]
        probs = F.softmax(logits, dim=1)                            # [N,C]
        k = min(k, probs.shape[1])
        top_p, top_i = torch.topk(probs, k=k, dim=1)                # [N,k], [N,k]

    results: List[List[Dict[str, Any]]] = []
    for p_row, i_row in zip(top_p.cpu(), top_i.cpu()):
        entries = []
        for p, idx in zip(p_row.tolist(), i_row.tolist()):
            label = classes[idx] if (classes and 0 <= idx < len(classes)) else f"class_{idx}"
            entries.append({"label": label, "prob": float(p), "class_id": int(idx)})
        results.append(entries)
    return results

@timeit
# --- remplacer votre analyser_image par ceci ---
@timeit
def analyser_image(image_path: str,
                   yolo_model,
                   resnet_model,
                   classes: Optional[List[str]],
                   device_resnet: str,
                   conf: float = 0.25,
                   iou: float = 0.45,
                   resnet_img_size: int = 224,
                   topk: int = 3):
    ensure_exists(image_path, "image")
    img = Image.open(image_path).convert("RGB")

    # 1) Détection (tous objets) puis filtrage avion
    detections = yolo_detect(yolo_model, image_path, conf, iou)
    detections = filter_airplane_detections(detections)

    if len(detections) == 0:
        return no_airplane_result()

    # 2) Rognage + classification Top-k
    crops = [crop_image(img, det["bbox"]) for det in detections]
    topk_per_crop = classify_topk(resnet_model, device_resnet, crops, classes, resnet_img_size, k=topk)

    # 3) Fusion résultats
    outputs = []
    for det, topk_list in zip(detections, topk_per_crop):
        outputs.append({
            "coordonnees": det["bbox"],
            "confiance_detection": det["score"],
            "label_yolo": det["label"],
            "topk": topk_list
        })

    return {
        "contains_airplane": True,
        "message": "Avion détecté",
        "results": outputs
    }


# =========================
# Main
# =========================
def main():
    device_yolo, device_resnet = get_device_labels()
    log(f"🔧 Configuration matérielle - YOLO: {device_yolo}, ResNet: {device_resnet}")
    try:
        # Chargement YOLO (CPU)
        if MODEL_YOLO_PATH.lower().endswith(".pt") or MODEL_YOLO_PATH.lower().endswith(".onnx"):
            yolo_model = load_yolo(MODEL_YOLO_PATH, device=device_yolo)
        else:
            raise ValueError("Fournissez un modèle YOLO .pt ou .onnx valide.")

        # Chargement ResNet (MPS si dispo)
        log("🔄 Chargement du modèle ResNet...")
        resnet_model, classes = load_resnet_autodetect(MODEL_RESNET_PATH, CLASSES, device=device_resnet)

        if classes is None:
            # Si aucune liste fournie ni trouvée, on crée des étiquettes numériques
            out_dim = resnet_model.fc.out_features
            classes = [f"class_{i}" for i in range(out_dim)]
            log(f"ℹ️ Aucune liste de classes fournie/trouvée, on utilise {out_dim} labels génériques.")

        # --- dans main(), remplacer le bloc inference/affichage par ceci ---
        # Étape 0) Garde‑fou: vérifie que l'image contient bien un avion
        log("🔍 Vérification: l'image contient-elle un avion ?")
        if not image_contains_airplane(yolo_model, IMAGE_PATH, conf=YOLO_CONF_THRES, iou=YOLO_IOU_THRES):
            res = no_airplane_result()
            log(f"🟡 {res['message']}")
            return  # on sort proprement

        # Inference (avec Top-3) — ici on est sûr qu'au moins une détection avion existe
        res = analyser_image(
            IMAGE_PATH,
            yolo_model,
            resnet_model,
            classes,
            device_resnet,
            conf=YOLO_CONF_THRES,
            iou=YOLO_IOU_THRES,
            resnet_img_size=RESNET_INPUT_SIZE,
            topk=3
        )

        if not res.get("contains_airplane", False) or len(res.get("results", [])) == 0:
            log("🟡 Aucun avion détecté")
            return

        log("\n📋 Résultats:")
        for i, r in enumerate(res["results"], 1):
            x1, y1, x2, y2 = r["coordonnees"]
            log(f"\n🛩️ Objet {i}:")
            log(f"   Coordonnées: ({x1}, {y1}, {x2}, {y2})")
            log(f"   Confiance YOLO: {r['confiance_detection']:.3f}")
            for rank, cand in enumerate(r["topk"], 1):
                log(f"   #{rank} {cand['label']} — p={cand['prob']:.4f} (id={cand['class_id']})")


    except Exception as e:
        # On tente d’être explicite sur les erreurs fréquentes
        msg = str(e)
        if "Missing key(s) in state_dict" in msg or "size mismatch" in msg:
            log("❌ Erreur fatale: ❌ Erreur dans analyser_image: ❌ Erreur chargement ResNet: " + msg)
        elif "ultralytics n'est pas installé" in msg:
            log("❌ Ultralytics manquant. Installez-le: pip install ultralytics")
        else:
            log("❌ Erreur fatale: " + msg)

if __name__ == "__main__":
    main()
