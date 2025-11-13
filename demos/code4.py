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
from PIL import Image, ImageDraw

# ============================================================
# Option GUI
# ============================================================
try:
    import gradio as gr
    _HAS_GRADIO = True
except Exception:
    _HAS_GRADIO = False

# ============================================================
# Ultralytics (YOLO)
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
MODEL_RESNET_PATH = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis-1/demos/resnet_model_optimized.pth"

# Dossier d'entraînement prioritaire (un sous-dossier par classe)
TRAIN_DIR = "/Users/a.cariven/Documents/USAFA/comp scien/crop"
DISCOVER_CLASSES_FROM_DIR = True      # True = impose les classes depuis TRAIN_DIR
NATURAL_SORT_CLASSES = False          # True = tri naturel (A10 < A320)
MIN_IMAGES_PER_CLASS = 1              # ignore les dossiers avec < N images

# Seuils YOLO
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45
YOLO_IMAGE_SIZE = 960  # plus grand que 640 pour avions

# ResNet
RESNET_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Extensions valides
_VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# ============================================================
# Utils
# ============================================================
def log(*a, **k):
    print(*a, **k, flush=True)

def timeit(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        dt = (time.time() - t0) * 1000
        log(f"⏱️ {fn.__name__} terminé en {dt:.1f} ms")
        return out
    return wrapper

def ensure_exists(path: str, kind: str = "fichier"):
    if not Path(path).exists():
        raise FileNotFoundError(f"Le {kind} '{path}' est introuvable.")

def get_device_labels():
    device_yolo = "cpu"  # imposé
    device_resnet = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return device_yolo, device_resnet

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def _count_images(d: Path, min_images: int) -> int:
    n = 0
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in _VALID_IMG_EXTS:
            n += 1
            if n >= min_images:
                break
    return n

def discover_classes_from_dir(train_dir: str,
                              include_hidden: bool = False,
                              natural_sort: bool = False,
                              min_images_per_class: int = 1) -> List[str]:
    root = Path(train_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dossier d'entraînement introuvable: {train_dir}")
    classes = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not include_hidden and d.name.startswith("."):
            continue
        if _count_images(d, min_images_per_class) < min_images_per_class:
            continue
        classes.append(d.name)
    if natural_sort:
        classes.sort(key=_natural_key)
    else:
        classes.sort()
    return classes

# ============================================================
# Transforms ResNet
# ============================================================
def build_resnet_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ============================================================
# Chargeur ResNet auto-détectant (strict, basé sur ton bloc)
# ============================================================
def _strip_prefix(state_dict, prefixes=("module.", "model.")):
    new_sd = {}
    for k, v in state_dict.items():
        k2 = k
        for p in prefixes:
            if k2.startswith(p):
                k2 = k2[len(p):]
        new_sd[k2] = v
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
    layout = _infer_layout(keys)  # e.g., (2,2,2,2) / (3,4,6,3) / (3,8,36,3)

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
    else:
        if layout == (2,2,2,2):
            model = models.resnet18(weights=None)
        elif layout == (3,4,6,3):
            model = models.resnet34(weights=None)
        else:
            model = models.resnet34(weights=None)

    in_feats = model.fc.in_features
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
def load_resnet_autodetect(model_path: str, classes_hint: Optional[List[str]], device: str):
    ensure_exists(model_path, "checkpoint ResNet")
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict, classes_in_ckpt = _extract_state_dict(ckpt)

    num_classes_hint = len(classes_hint) if isinstance(classes_hint, list) else None
    model, state_dict, num_classes = _build_matching_resnet(state_dict, num_classes_hint)

    if classes_in_ckpt is not None and isinstance(classes_in_ckpt, (list, tuple)):
        if classes_hint is None:
            classes_hint = list(classes_in_ckpt)
        elif len(classes_hint) != len(classes_in_ckpt):
            log(f"⚠️ Liste classes fournie ({len(classes_hint)}) ≠ checkpoint ({len(classes_in_ckpt)}). "
                f"On garde la tête dimension {num_classes} du ckpt pour chargement strict.")

    log(f"ℹ️ ResNet détecté: "
        f"{'Bottleneck' if model.layer1[0].expansion==4 else 'BasicBlock'} | "
        f"fc.in={model.fc.in_features} | classes={num_classes}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model, classes_hint

# ============================================================
# YOLO
# ============================================================
@timeit
def load_yolo(path: str, device: str = "cpu"):
    if YOLO is None:
        raise RuntimeError("ultralytics n'est pas installé. pip install ultralytics")
    ensure_exists(path, "modèle YOLO")
    return YOLO(path)

# Détection flexible: accepte PIL.Image ou chemin
def yolo_detect(yolo_model, image: Union[str, Image.Image], conf: float, iou: float) -> List[Dict[str, Any]]:
    results = yolo_model.predict(
        source=image,
        device="cpu",
        conf=conf,
        iou=iou,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False
    )
    dets: List[Dict[str, Any]] = []
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

# ============================================================
# Image utils + rendu
# ============================================================
def clamp_bbox(b: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return int(x1), int(y1), int(x2), int(y2)

def crop_image(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    w, h = pil_img.size
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    return pil_img.crop((x1, y1, x2, y2)).convert("RGB")

def draw_boxes(img: Image.Image, dets: List[Dict[str, Any]]) -> Image.Image:
    out = img.copy()
    drw = ImageDraw.Draw(out)
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = d["bbox"]
        drw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        txt = f"{i} | conf={d.get('score',0):.2f}"
        drw.text((x1+4, y1+4), txt, fill=(255, 0, 0))
    return out

# ============================================================
# Classification Top‑k (ton bloc)
# ============================================================
@torch.inference_mode()
def classify_topk(model_resnet,
                  device: str,
                  crops: List[Image.Image],
                  classes: Optional[List[str]],
                  img_size: int = 224,
                  k: int = 3) -> List[List[Dict[str, Any]]]:
    if len(crops) == 0:
        return []
    tfm = build_resnet_transform(img_size)
    batch = torch.stack([tfm(im) for im in crops]).to(device)   # [N,3,H,W]
    logits = model_resnet(batch)                                # [N,C]
    probs = F.softmax(logits, dim=1)                            # [N,C]
    k = min(max(1, k), probs.shape[1])
    top_p, top_i = torch.topk(probs, k=k, dim=1)                # [N,k], [N,k]

    results: List[List[Dict[str, Any]]] = []
    for p_row, i_row in zip(top_p.cpu(), top_i.cpu()):
        entries = []
        for p, idx in zip(p_row.tolist(), i_row.tolist()):
            label = classes[idx] if (classes and 0 <= idx < len(classes)) else f"class_{idx}"
            entries.append({"label": label, "prob": float(p), "class_id": int(idx)})
        results.append(entries)
    return results

# ============================================================
# Analyse (accepte chemin ou PIL)
# ============================================================
@timeit
def analyser_image(
    image: Union[str, Image.Image],
    yolo_model,
    resnet_model,
    classes: Optional[List[str]],
    device_resnet: str,
    conf: float = YOLO_CONF_THRES,
    iou: float = YOLO_IOU_THRES,
    resnet_img_size: int = RESNET_INPUT_SIZE,
    topk: int = 3,
    return_drawn: bool = True
) -> Dict[str, Any]:

    if isinstance(image, str):
        ensure_exists(image, "image")
        pil = Image.open(image).convert("RGB")
        detect_input = image
    else:
        pil = image.convert("RGB")
        detect_input = pil

    detections = yolo_detect(yolo_model, detect_input, conf, iou)

    crops, topk_per_crop = [], []
    if len(detections) > 0:
        crops = [crop_image(pil, det["bbox"]) for det in detections]
        topk_per_crop = classify_topk(resnet_model, device_resnet, crops, classes, resnet_img_size, k=topk)

    annotated = draw_boxes(pil, detections) if return_drawn else None

    results_struct = []
    for det, topk_list in zip(detections, topk_per_crop):
        x1, y1, x2, y2 = det["bbox"]
        results_struct.append({
            "coordonnees": (x1, y1, x2, y2),
            "confiance_detection": det["score"],
            "topk": topk_list
        })

    return {
        "annotated": annotated,
        "detections": detections,
        "results": results_struct,
        "crops": crops
    }

# ============================================================
# Prépare classes utilisateur depuis TRAIN_DIR (PRIORITAIRE)
# ============================================================
def get_user_classes() -> Optional[List[str]]:
    if not DISCOVER_CLASSES_FROM_DIR:
        return None
    try:
        classes = discover_classes_from_dir(
            TRAIN_DIR,
            include_hidden=False,
            natural_sort=NATURAL_SORT_CLASSES,
            min_images_per_class=MIN_IMAGES_PER_CLASS
        )
        if len(classes) == 0:
            log(f"⚠️ Aucune classe trouvée dans {TRAIN_DIR}. Fallback ckpt.")
            return None
        log(f"📁 Classes découvertes dans TRAIN_DIR ({len(classes)}): {classes}")
        return classes
    except Exception as e:
        log(f"⚠️ Découverte depuis TRAIN_DIR échouée ({type(e).__name__}: {e}). Fallback ckpt.")
        return None

# ============================================================
# Sanity checks
# ============================================================
def sanity_checks(resnet_model: nn.Module, classes: Optional[List[str]], device: str):
    out_dim = resnet_model.fc.out_features
    log(f"✅ fc.out={out_dim} | #classes_hint={0 if classes is None else len(classes)} | device={device}")
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, RESNET_INPUT_SIZE, RESNET_INPUT_SIZE, dtype=torch.float32, device=device)
        logits = resnet_model(dummy)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Logits non finis (NaN/Inf).")

# ============================================================
# CLI
# ============================================================
def run_cli(args):
    device_yolo, device_resnet = get_device_labels()
    log(f"🔧 Devices — YOLO: {device_yolo}, ResNet: {device_resnet}")

    yolo = load_yolo(MODEL_YOLO_PATH, device=device_yolo)

    user_classes = get_user_classes()
    resnet, classes = load_resnet_autodetect(MODEL_RESNET_PATH, user_classes, device=device_resnet)
    if classes is None:
        classes = [f"class_{i}" for i in range(resnet.fc.out_features)]
        log(f"ℹ️ Aucune liste de classes fournie/trouvée, on utilise {len(classes)} labels génériques.")
    sanity_checks(resnet, classes, device_resnet)

    ensure_exists(args.image, "image")
    out = analyser_image(
        image=args.image,
        yolo_model=yolo,
        resnet_model=resnet,
        classes=classes,
        device_resnet=device_resnet,
        conf=args.conf,
        iou=args.iou,
        resnet_img_size=RESNET_INPUT_SIZE,
        topk=args.topk,
        return_drawn=True
    )

    dets = out["results"]
    if len(dets) == 0:
        log("⚠️ Aucun objet détecté par YOLO.")
    else:
        log("\n📋 Résultats:")
        for i, r in enumerate(dets, 1):
            x1, y1, x2, y2 = r["coordonnees"]
            log(f"\n🛩️ Objet {i}: bbox=({x1},{y1},{x2},{y2}) | conf={r['confiance_detection']:.3f}")
            for rank, cand in enumerate(r["topk"], 1):
                log(f"   #{rank} {cand['label']} — p={cand['prob']:.4f} (id={cand['class_id']})")

    if args.save:
        out_path = Path(args.save)
        out["annotated"].save(out_path)
        log(f"💾 Image annotée sauvegardée: {out_path}")

# ============================================================
# Gradio
# ============================================================
def build_ui():
    if not _HAS_GRADIO:
        raise RuntimeError("Gradio n'est pas installé. pip install gradio")

    device_yolo, device_resnet = get_device_labels()
    yolo = load_yolo(MODEL_YOLO_PATH, device=device_yolo)

    user_classes = get_user_classes()
    resnet, classes = load_resnet_autodetect(MODEL_RESNET_PATH, user_classes, device=device_resnet)
    if classes is None:
        classes = [f"class_{i}" for i in range(resnet.fc.out_features)]
        log(f"ℹ️ Aucune liste de classes fournie/trouvée, on utilise {len(classes)} labels génériques.")
    sanity_checks(resnet, classes, device_resnet)

    def predict(pil_image: Optional[Image.Image], conf: float, iou: float, topk: int):
        if pil_image is None:
            return None, "Veuillez uploader une image.", None
        try:
            out = analyser_image(
                image=pil_image,
                yolo_model=yolo,
                resnet_model=resnet,
                classes=classes,
                device_resnet=device_resnet,
                conf=conf,
                iou=iou,
                resnet_img_size=RESNET_INPUT_SIZE,
                topk=topk,
                return_drawn=True
            )
            dets = out["results"]
            if len(dets) == 0:
                return out["annotated"], "Aucun objet détecté", None
            lines = []
            for i, r in enumerate(dets, 1):
                x1, y1, x2, y2 = r["coordonnees"]
                lines.append(f"🛩️ Objet {i} - bbox=({x1},{y1},{x2},{y2}) | conf={r['confiance_detection']:.3f}")
                for rk, cand in enumerate(r["topk"], 1):
                    lines.append(f"   #{rk} {cand['label']} — p={cand['prob']:.4f} (id={cand['class_id']})")
            text = "\n".join(lines)
            main_crop = out["crops"][0] if len(out["crops"]) else None
            return out["annotated"], text, main_crop
        except Exception as e:
            return pil_image, f"❌ Exception: {type(e).__name__}: {e}", None

    with gr.Blocks(title="Détecteur d'avions + Top‑k") as demo:
        gr.Markdown("## ✈️ Détection d’avions (YOLO CPU) + Classification (ResNet) Top‑k")
        gr.Markdown("Les classes affichées proviennent de TRAIN_DIR si disponibles.")
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Image", image_mode="RGB")
                conf = gr.Slider(0.05, 0.9, value=YOLO_CONF_THRES, step=0.05, label="Seuil confiance YOLO")
                iou = gr.Slider(0.1, 0.9, value=YOLO_IOU_THRES, step=0.05, label="Seuil IoU YOLO")
                topk = gr.Slider(1, 5, value=3, step=1, label="Top‑k ResNet")
                btn = gr.Button("Analyser")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Image annotée")
                out_txt = gr.Textbox(label="Résultats", lines=12)
                out_crop = gr.Image(type="pil", label="Crop du 1er objet")
        btn.click(fn=predict, inputs=[in_img, conf, iou, topk], outputs=[out_img, out_txt, out_crop])
        in_img.change(fn=predict, inputs=[in_img, conf, iou, topk], outputs=[out_img, out_txt, out_crop])
    return demo

# ============================================================
# Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Détection (YOLO) + classification (ResNet) Top‑k")
    ap.add_argument("--image", type=str, help="Chemin d'une image pour la version CLI")
    ap.add_argument("--conf", type=float, default=YOLO_CONF_THRES, help="Seuil confiance YOLO")
    ap.add_argument("--iou", type=float, default=YOLO_IOU_THRES, help="Seuil IoU YOLO")
    ap.add_argument("--topk", type=int, default=3, help="Top‑k pour ResNet")
    ap.add_argument("--save", type=str, default="", help="Chemin de sortie pour l'image annotée")
    ap.add_argument("--gradio", action="store_true", help="Lancer l'interface Gradio")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.gradio:
        demo = build_ui()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    else:
        if not args.image:
            print("Veuillez fournir --image fichier.jpg (ou utilisez --gradio).")
            sys.exit(1)
        run_cli(args)

if __name__ == "__main__":
    main()
