#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Option GUI
# ============================================================
try:
    import gradio as gr
    _HAS_GRADIO = True
except Exception:
    _HAS_GRADIO = False

# ============================================================
# YOLO (ultralytics)
# ============================================================
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False

# ============================================================
# Config (adaptez ici)
# ============================================================
MODEL_YOLO_PATH = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis-1/demos/yolov8n.pt"
MODEL_RESNET_PATH = "/Users/a.cariven/Documents/USAFA/resnet_model_optimized.pth"

# Dossier d'entraînement prioritaire (un sous-dossier par classe)
TRAIN_DIR = "/Users/a.cariven/Documents/USAFA/comp scien/crop"
DISCOVER_CLASSES_FROM_DIR = True      # True = impose les classes depuis TRAIN_DIR
NATURAL_SORT_CLASSES = False          # True = tri naturel (A10 < A320)
MIN_IMAGES_PER_CLASS = 1              # ignore les dossiers avec < N images

# Seuils YOLO par défaut
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45
YOLO_IMAGE_SIZE = 960   # plus grand pour petits avions
YOLO_MAX_DET = 100
YOLO_TTA = True         # Test-Time Augmentation
YOLO_AGNOSTIC_NMS = True
# ------------------------------------------------------------
# Filtrage initial: ne garder que les avions (optionnel)
ONLY_KEEP_AIRPLANES = True   # True = ne garder que les détections de classe "airplane"
AIRPLANE_CLASS_ID = 4        # COCO class id pour "airplane" / "aeroplane"
# ------------------------------------------------------------


# ResNet
RESNET_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Divers
DRAW_FONT_SIZE = 16
_VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ============================================================
# Gestion des devices
# ============================================================
def get_device_labels():
    device_yolo = "cpu"  # imposé
    device_resnet = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return device_yolo, device_resnet

# ============================================================
# Utils
# ============================================================
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

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# ============================================================
# Découverte des classes
# ============================================================
def discover_classes_from_dir(root: str, min_images: int = 1, natural_sort: bool = False) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    classes = []
    for d in os.listdir(root):
        if d.startswith("."):
            continue
        p = os.path.join(root, d)
        if os.path.isdir(p):
            # compte d’images valides
            n = 0
            for f in os.listdir(p):
                if os.path.splitext(f)[1].lower() in _VALID_IMG_EXTS:
                    n += 1
            if n >= min_images:
                classes.append(d)
    if natural_sort:
        classes.sort(key=natural_key)
    else:
        classes.sort()
    return classes

# ============================================================
# Pré-traitements ResNet
# ============================================================
def build_resnet_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ============================================================
# Chargeur ResNet auto-détectant
# ============================================================
def _strip_prefix(state_dict, prefixes=("module.", "model.")):
    new_sd = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        new_sd[k] = v
    return new_sd

def _is_bottleneck(state_keys):
    return any("layer1.0.conv3.weight" in k for k in state_keys)

def _infer_layout(state_keys):
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

    is_bneck = _is_bottleneck(keys)
    layout = _infer_layout(keys)  # e.g. (2,2,2,2), (3,4,6,3), (3,8,36,3)

    # nb classes d’après fc.weight si dispo
    num_classes_ckpt = None
    if "fc.weight" in state_dict and isinstance(state_dict["fc.weight"], torch.Tensor):
        num_classes_ckpt = state_dict["fc.weight"].shape[0]
    num_classes = num_classes_hint or num_classes_ckpt or 1000

    if is_bneck:
        if layout == (3,4,6,3):
            model = models.resnet50(weights=None)
        elif layout == (3,4,23,3):
            model = models.resnet101(weights=None)
        elif layout == (3,8,36,3):
            model = models.resnet152(weights=None)
        else:
            model = models.resnet50(weights=None)
        in_feats = model.fc.in_features  # 2048
    else:
        if layout == (2,2,2,2):
            model = models.resnet18(weights=None)
        elif layout == (3,4,6,3):
            model = models.resnet34(weights=None)
        else:
            model = models.resnet34(weights=None)
        in_feats = model.fc.in_features  # 512

    model.fc = nn.Linear(in_feats, num_classes)
    return model, state_dict, num_classes

def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"], ckpt_obj.get("classes", None)
        for k in ("model", "net"):
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

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model, classes

# ============================================================
# Chargement YOLO (CPU)
# ============================================================
@timeit
def load_yolo(model_path: str, device: str = "cpu"):
    if YOLO is None:
        raise RuntimeError("ultralytics n'est pas installé. Installez-le: pip install ultralytics")
    ensure_exists(model_path, "modèle YOLO")
    model = YOLO(model_path)
    return model

# ============================================================
# Inference helpers
# ============================================================
def yolo_detect(yolo_model, image: Union[str, Image.Image], conf: float, iou: float,
                imgsz: int = YOLO_IMAGE_SIZE,
                tta: bool = YOLO_TTA,
                agnostic_nms: bool = YOLO_AGNOSTIC_NMS,
                max_det: int = YOLO_MAX_DET) -> List[Dict[str, Any]]:
    results = yolo_model.predict(
        source=image,
        device="cpu",
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        augment=tta,
        agnostic_nms=agnostic_nms,
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
        cls_id = int(cls_ids[i]) if cls_ids is not None else -1
        dets.append({
            "bbox": (x1, y1, x2, y2),
            "score": float(scores[i]),
            "cls_id": cls_id
        })

    # Filtrage optionnel : ne garder que la classe "airplane"
    if ONLY_KEEP_AIRPLANES:
        before_n = len(dets)
        dets = [d for d in dets if d.get("cls_id", -1) == AIRPLANE_CLASS_ID]
        after_n = len(dets)
        log(f"ℹ️ Filtrage avions: {after_n}/{before_n} détections conservées (classe id={AIRPLANE_CLASS_ID})")

    return dets


def crop_image(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
    return pil_img.crop((x1, y1, x2, y2))

def build_resnet_batch(crops: List[Image.Image], img_size: int) -> torch.Tensor:
    tfm = build_resnet_transform(img_size)
    return torch.stack([tfm(im) for im in crops])  # [N,3,H,W]

def classify_topk(model_resnet,
                  device: str,
                  crops: List[Image.Image],
                  classes: Optional[List[str]],
                  img_size: int = 224,
                  k: int = 3) -> List[List[Dict[str, Any]]]:
    if len(crops) == 0:
        return []
    with torch.no_grad():
        batch = build_resnet_batch(crops, img_size).to(device)   # [N,3,H,W]
        logits = model_resnet(batch)                              # [N,C]
        probs = F.softmax(logits, dim=1)                          # [N,C]
        k = min(k, probs.shape[1])
        top_p, top_i = torch.topk(probs, k=k, dim=1)              # [N,k], [N,k]
    results: List[List[Dict[str, Any]]] = []
    for p_row, i_row in zip(top_p.cpu(), top_i.cpu()):
        entries = []
        for p, idx in zip(p_row.tolist(), i_row.tolist()):
            label = classes[idx] if (classes and 0 <= idx < len(classes)) else f"class_{idx}"
            entries.append({"label": label, "prob": float(p), "class_id": int(idx)})
        results.append(entries)
    return results

@timeit
@timeit
def analyser_image(image: Union[str, Image.Image, None],
                   yolo_model,
                   resnet_model,
                   classes: Optional[List[str]],
                   device_resnet: str,
                   conf: float = YOLO_CONF_THRES,
                   iou: float = YOLO_IOU_THRES,
                   resnet_img_size: int = RESNET_INPUT_SIZE,
                   topk: int = 3,
                   imgsz: int = YOLO_IMAGE_SIZE):
    # Cas UI: pas encore d'image
    if image is None:
        return [], None, None

    if isinstance(image, str):
        ensure_exists(image, "image")
        pil = Image.open(image).convert("RGB")
        yolo_src = image
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
        yolo_src = pil
    else:
        # Type inattendu
        return [], None, None

    detections = yolo_detect(yolo_model, yolo_src, conf, iou, imgsz=imgsz)
    if len(detections) == 0:
        # Retourne bien un PIL.Image pour Gradio, même si vide de détections
        return [], pil, None

    crops = [crop_image(pil, det["bbox"]) for det in detections]
    topk_per_crop = classify_topk(resnet_model, device_resnet, crops, classes, resnet_img_size, k=topk)

    outputs = []
    for det, topk_list in zip(detections, topk_per_crop):
        outputs.append({
            "coordonnees": det["bbox"],
            "confiance_detection": det["score"],
            "topk": topk_list
        })

    annotated, first_crop = draw_annotations(pil, outputs)
    return outputs, annotated, first_crop

# ============================================================
# Dessin
# ============================================================
def get_font(size: int = DRAW_FONT_SIZE):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def draw_annotations(img: Image.Image, outputs: List[Dict[str, Any]]) -> Tuple[Image.Image, Optional[Image.Image]]:
    # Toujours retourner un PIL.Image, jamais draw.im
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    font = get_font(DRAW_FONT_SIZE)

    first_crop = None
    for i, out in enumerate(outputs):
        x1, y1, x2, y2 = out["coordonnees"]
        score = out["confiance_detection"]
        if out["topk"]:
            label = out["topk"][0]["label"]
            p = out["topk"][0]["prob"]
        else:
            label, p = "?", 0.0

        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        text = f"{label} {p:.2f} | det {score:.2f}"
        # Largeur/hauteur du fond texte (compatibilité Pillow)
        try:
            # Pillow ≥ 8.0
            tw = draw.textlength(text, font=font)
            th = DRAW_FONT_SIZE + 6
        except Exception:
            # Fallback universel
            bx0, by0, bx1, by1 = draw.textbbox((x1, y1), text, font=font)
            tw, th = (bx1 - bx0), (by1 - by0)

        bx2 = min(x1 + int(tw) + 10, canvas.width)
        by2 = y1 + int(th)
        draw.rectangle([x1, y1, bx2, by2], fill=(0, 255, 0))
        draw.text((x1 + 5, y1 + 3), text, fill=(0, 0, 0), font=font)

        if i == 0:
            first_crop = crop_image(canvas, (x1, y1, x2, y2))
    return canvas, first_crop


# ============================================================
# Wrappers CLI / UI
# ============================================================
def format_results_text(outputs: List[Dict[str, Any]]) -> str:
    if not outputs:
        return "⚠️ Aucun objet détecté."
    lines = ["📋 Résultats:"]
    for i, r in enumerate(outputs, 1):
        x1, y1, x2, y2 = r["coordonnees"]
        lines.append(f"\n🛩️ Objet {i}: bbox=({x1},{y1},{x2},{y2}) | conf_det={r['confiance_detection']:.3f}")
        for rank, cand in enumerate(r["topk"], 1):
            lines.append(f"   #{rank} {cand['label']} — p={cand['prob']:.4f} (id={cand['class_id']})")
    return "\n".join(lines)
def predict(pil_img, conf, iou, topk, imgsz):
    # Lazy singleton for heavy models
    if not hasattr(predict, "_rt"):
        predict._rt = build_runtime()
    yolo_model, resnet_model, classes, device_resnet = predict._rt

    # No image yet: return placeholders that Gradio accepts
    if pil_img is None:
        return None, "📥 Importez une image pour lancer l'analyse.", None

    outputs, annotated, first_crop = analyser_image(
        pil_img, yolo_model, resnet_model, classes, device_resnet,
        conf=conf, iou=iou, resnet_img_size=RESNET_INPUT_SIZE, topk=topk, imgsz=imgsz
    )
    return annotated, format_results_text(outputs), first_crop

def build_runtime():
    device_yolo, device_resnet = get_device_labels()
    log(f"🔧 Matériel — YOLO: {device_yolo}, ResNet: {device_resnet}")

    # Découverte classes
    classes = None
    if DISCOVER_CLASSES_FROM_DIR:
        classes = discover_classes_from_dir(TRAIN_DIR, MIN_IMAGES_PER_CLASS, NATURAL_SORT_CLASSES)
        if classes:
            log(f"📁 Classes découvertes ({len(classes)}): {classes[:10]}{' ...' if len(classes)>10 else ''}")
        else:
            log("⚠️ Aucune classe découverte depuis TRAIN_DIR; on tentera depuis le checkpoint.")

    # Loaders
    yolo_model = load_yolo(MODEL_YOLO_PATH, device=device_yolo)
    log("🔄 Chargement ResNet...")
    resnet_model, classes = load_resnet_autodetect(MODEL_RESNET_PATH, classes, device=device_resnet)

    # Fallback classes si toujours None
    if classes is None:
        out_dim = resnet_model.fc.out_features
        classes = [f"class_{i}" for i in range(out_dim)]
        log(f"ℹ️ Classes génériques utilisées: {out_dim}")

    return yolo_model, resnet_model, classes, device_resnet
def run_cli(args):
    yolo_model, resnet_model, classes, device_resnet = build_runtime()
    outputs, annotated, first_crop = analyser_image(
        args.image, yolo_model, resnet_model, classes, device_resnet,
        conf=args.conf, iou=args.iou, resnet_img_size=RESNET_INPUT_SIZE, topk=args.topk, imgsz=args.imgsz
    )
    print(format_results_text(outputs))
    if args.save:
        if annotated is None:
            print("⚠️ Rien à sauvegarder (pas d'image fournie ou erreur).")
        else:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            annotated.save(args.save)
            print(f"💾 Image annotée sauvegardée: {args.save}")



def build_ui():
    if not _HAS_GRADIO:
        raise RuntimeError("Gradio n’est pas installé. pip install gradio")
    with gr.Blocks(title="YOLO + ResNet") as demo:
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Image d’entrée")
                conf = gr.Slider(0.0, 1.0, value=YOLO_CONF_THRES, step=0.01, label="Confiance YOLO")
                iou = gr.Slider(0.0, 1.0, value=YOLO_IOU_THRES, step=0.01, label="IoU YOLO")
                imgsz = gr.Slider(320, 1536, value=YOLO_IMAGE_SIZE, step=32, label="Taille d’inférence YOLO")
                topk = gr.Slider(1, 5, value=3, step=1, label="Top‑k ResNet")
                btn = gr.Button("Analyser")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Image annotée")
                out_txt = gr.Textbox(label="Résultats", lines=14)
                out_crop = gr.Image(type="pil", label="Crop du 1er objet")

        # wrapper local (évite dépendance à l’ordre si vous réorganisez plus tard)
        def on_click(pil_img, conf, iou, topk, imgsz):
            return predict(pil_img, conf, iou, topk, imgsz)

        btn.click(fn=on_click, inputs=[in_img, conf, iou, topk, imgsz],
                  outputs=[out_img, out_txt, out_crop])
        in_img.change(fn=on_click, inputs=[in_img, conf, iou, topk, imgsz],
                      outputs=[out_img, out_txt, out_crop])
    return demo

        # Wrapper local pour éviter le NameError tant que predict n'est pas encore défini
      

# ============================================================
# Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Détection (YOLO) + classification (ResNet) Top‑k")
    ap.add_argument("--image", type=str, help="Chemin d'une image pour la version CLI")
    ap.add_argument("--conf", type=float, default=YOLO_CONF_THRES, help="Seuil confiance YOLO")
    ap.add_argument("--iou", type=float, default=YOLO_IOU_THRES, help="Seuil IoU YOLO")
    ap.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE, help="Taille d’inférence YOLO")
    ap.add_argument("--topk", type=int, default=3, help="Top‑k pour ResNet")
    ap.add_argument("--save", type=str, default="", help="Chemin de sortie pour l'image annotée (CLI)")
    ap.add_argument("--gradio", action="store_true", help="Lancer l'interface Gradio")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.gradio:
        if not _HAS_GRADIO:
            print("Gradio non installé. Faites: pip install gradio", file=sys.stderr)
            sys.exit(2)
        # build_ui doit être défini à ce stade
        demo = build_ui()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
        return

    # CLI
    if not args.image:
        print("Veuillez fournir --image fichier.jpg (ou utilisez --gradio).", file=sys.stderr)
        sys.exit(1)
    run_cli(args)

if __name__ == "__main__":
    main()
