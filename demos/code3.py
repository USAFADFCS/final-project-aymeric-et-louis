#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageDraw
import gradio as gr
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

# Si vous avez un dossier avec une sous‑dossier par classe, laissez None pour forcer sidecar/ckpt
USER_CLASSES: Optional[List[str]] = None
# Exemple:
# data_dir = "/path/crop"
# USER_CLASSES = sorted([d for d in os.listdir(data_dir) if (Path(data_dir)/d).is_dir() and not d.startswith(".")])

# Seuils YOLO
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45
YOLO_IMAGE_SIZE = 960  # plus grand que 640 pour avions

# ResNet
RESNET_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Étiquettes reconnues comme “avion”
AIRPLANE_LABELS = {"airplane", "aeroplane", "aircraft", "plane"}

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

def get_devices() -> tuple[str, str]:
    device_yolo = "cpu"  # YOLO CPU par défaut (stable partout)
    device_resnet = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return device_yolo, device_resnet

def ensure_exists(path: str, kind: str = "fichier"):
    if not Path(path).exists():
        raise FileNotFoundError(f"Le {kind} '{path}' est introuvable.")

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
# Chargement classes
# ============================================================
def _load_classes_from_sidecar(model_path: str) -> Optional[List[str]]:
    p = Path(model_path)
    candidates = [
        p.with_suffix(".classes.json"),
        p.parent / "classes.json",
    ]
    for fp in candidates:
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
                return data["classes"]
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
    return None

def _classes_from_ckpt(ckpt: dict) -> Optional[List[str]]:
    if not isinstance(ckpt, dict):
        return None
    # direct
    if "classes" in ckpt and isinstance(ckpt["classes"], list):
        return ckpt["classes"]
    # nested dicts
    for key in ["meta", "args", "hparams", "config"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            sub = ckpt[key]
            if "classes" in sub and isinstance(sub["classes"], list):
                return sub["classes"]
            if "class_to_idx" in sub and isinstance(sub["class_to_idx"], dict):
                items = sorted(sub["class_to_idx"].items(), key=lambda kv: kv[1])
                return [k for k, _ in items]
    # class_to_idx at root
    if "class_to_idx" in ckpt and isinstance(ckpt["class_to_idx"], dict):
        items = sorted(ckpt["class_to_idx"].items(), key=lambda kv: kv[1])
        return [k for k, _ in items]
    return None

# ============================================================
# Modèle ResNet
# ============================================================
def _clean_state_dict(state: dict) -> dict:
    new_state = {}
    for k, v in state.items():
        k2 = k
        for p in ["module.", "model."]:
            if k2.startswith(p):
                k2 = k2[len(p):]
        new_state[k2] = v
    return new_state

def _is_bottleneck(keys: List[str]) -> bool:
    # True pour resnet50/101/152 (présence conv3 dans le premier bloc de layer1)
    return any("layer1.0.conv3.weight" in k for k in keys)

def build_resnet_from_state(state: dict) -> nn.Module:
    keys = list(state.keys())
    backbone = models.resnet50(weights=None) if _is_bottleneck(keys) else models.resnet34(weights=None)
    return backbone

def load_resnet_autodetect(model_path: str,
                           user_classes: Optional[List[str]],
                           device: str = "cpu") -> tuple[nn.Module, List[str]]:
    ensure_exists(model_path, "checkpoint ResNet")
    sd = torch.load(model_path, map_location="cpu")

    if isinstance(sd, dict) and "state_dict" in sd:
        state = sd["state_dict"]
        ckpt_meta = sd
    else:
        state = sd
        ckpt_meta = sd if isinstance(sd, dict) else {}

    state = _clean_state_dict(state)
    model = build_resnet_from_state(state)

    # Résoudre les classes par priorité
    classes = user_classes or _load_classes_from_sidecar(model_path) or _classes_from_ckpt(ckpt_meta)
    if classes is None:
        out_dim = state["fc.weight"].shape[0] if "fc.weight" in state else model.fc.out_features
        classes = [f"class_{i}" for i in range(out_dim)]

    # Adapter la tête à len(classes)
    out_dim = len(classes)
    if model.fc.out_features != out_dim:
        model.fc = nn.Linear(model.fc.in_features, out_dim)

    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"ℹ️ ResNet chargé (manquantes={len(missing)}, inattendues={len(unexpected)})")

    model.eval().to(device)
    if device == "mps":
        model = model.float()  # FP32 recommandé
    return model, classes

# ============================================================
# YOLO
# ============================================================
def ensure_ultralytics():
    if not _HAS_YOLO:
        raise RuntimeError("Ultralytics n'est pas installé. Faites: pip install -U ultralytics")

def load_yolo(path: str):
    ensure_exists(path, "modèle YOLO")
    ensure_ultralytics()
    model = YOLO(path)
    return model

@timeit
def yolo_detect(yolo_model, image: Image.Image, conf: float, iou: float) -> List[Dict[str, Any]]:
    # Ultralytics accepte directement un PIL.Image
    results = yolo_model.predict(
        source=image,
        device="cpu",                # YOLO sur CPU
        conf=conf,
        iou=iou,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False
    )
    dets: List[Dict[str, Any]] = []
    if not results:
        return dets
    r0 = results[0]
    if not hasattr(r0, "boxes") or r0.boxes is None or len(r0.boxes) == 0:
        return dets
    names = r0.names if hasattr(r0, "names") else {}
    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    cls   = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes.cls is not None else None

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        cid = int(cls[i]) if cls is not None else -1
        label = names.get(cid, str(cid))
        dets.append({
            "bbox": (x1, y1, x2, y2),
            "score": float(confs[i]),
            "class_id": cid,
            "label": label
        })
    return dets

def filter_airplanes(dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [d for d in dets if str(d.get("label", "")).lower() in AIRPLANE_LABELS]

# ============================================================
# Image utils
# ============================================================
def clamp_bbox(b: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return int(x1), int(y1), int(x2), int(y2)

def crop_image(img: Image.Image, bbox: Tuple[int,int,int,int]) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    return img.crop((x1, y1, x2, y2)).convert("RGB")

def draw_boxes(img: Image.Image, dets: List[Dict[str, Any]]) -> Image.Image:
    out = img.copy()
    drw = ImageDraw.Draw(out)
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = d["bbox"]
        drw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        txt = f"{i}: {d.get('label','?')} {d.get('score',0):.2f}"
        drw.text((x1+4, y1+4), txt, fill=(255, 0, 0))
    return out

# ============================================================
# Classification Top‑k
# ============================================================
@torch.inference_mode()
def classify_topk(model: nn.Module,
                  device: str,
                  crops: List[Image.Image],
                  classes: List[str],
                  img_size: int = 224,
                  k: int = 3) -> List[List[Dict[str, Any]]]:
    if model is None or len(crops) == 0:
        return []
    tfm = build_resnet_transform(img_size)
    batch = torch.stack([tfm(c) for c in crops], dim=0)  # [N,3,H,W]

    # Dtypes
    if device == "mps":
        batch = batch.to(device, dtype=torch.float32)
    elif device == "cuda" and torch.cuda.is_available():
        batch = batch.to(device, dtype=torch.float32)  # FP32 sûr; FP16 possible si vous voulez
    else:
        batch = batch.to(device, dtype=torch.float32)

    model = model.to(device)
    model.eval()
    logits = model(batch)                 # [N,C]
    probs = F.softmax(logits, dim=1)
    top_p, top_i = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)

    res: List[List[Dict[str, Any]]] = []
    for p_row, i_row in zip(top_p.cpu(), top_i.cpu()):
        items = []
        for p, idx in zip(p_row.tolist(), i_row.tolist()):
            lbl = classes[idx] if 0 <= idx < len(classes) else f"class_{idx}"
            items.append({"label": lbl, "prob": float(p), "class_id": int(idx)})
        res.append(items)
    return res

# ============================================================
# Pipeline
# ============================================================
def analyze_image(pil_image: Image.Image,
                  yolo_model,
                  resnet_model: nn.Module,
                  classes: List[str],
                  device_resnet: str,
                  conf: float,
                  iou: float,
                  topk: int,
                  return_drawn: bool = True) -> Dict[str, Any]:
    # 1) YOLO
    dets_all = yolo_detect(yolo_model, pil_image, conf, iou)
    dets_air = filter_airplanes(dets_all)

    # 2) Classification si avions détectés
    topk_lists: List[List[Dict[str, Any]]] = []
    crops: List[Image.Image] = []
    if len(dets_air) > 0:
        crops = [crop_image(pil_image, d["bbox"]) for d in dets_air]
        topk_lists = classify_topk(resnet_model, device_resnet, crops, classes, RESNET_INPUT_SIZE, k=topk)

    # 3) Image annotée
    annotated = draw_boxes(pil_image, dets_air) if return_drawn else None

    # 4) Struct résultat
    results_struct = []
    for d, tk in zip(dets_air, topk_lists):
        x1, y1, x2, y2 = d["bbox"]
        results_struct.append({
            "coordonnees": (x1, y1, x2, y2),
            "confiance_detection": d["score"],
            "topk": tk
        })
    return {
        "annotated": annotated,
        "detections": dets_air,
        "results": results_struct,
        "crops": crops
    }

# ============================================================
# Sanity checks
# ============================================================
def sanity_checks(resnet_model: nn.Module, classes: List[str], device: str):
    log(f"✅ fc.out={resnet_model.fc.out_features} | #classes={len(classes)}")
    if resnet_model.fc.out_features != len(classes):
        raise RuntimeError("Incohérence tête fc ↔ nombre de classes. Corrigez l’ordre/nb des classes (classes.json).")
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)
        dummy = dummy.to(device, dtype=torch.float32 if device == "mps" else torch.float32)
        logits = resnet_model(dummy)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Logits non finis (NaN/Inf) — problème de dtype/device.")

# ============================================================
# CLI
# ============================================================
def run_cli(args):
    device_yolo, device_resnet = get_devices()
    log(f"🔧 Devices — YOLO: {device_yolo}, ResNet: {device_resnet}")

    yolo = load_yolo(MODEL_YOLO_PATH)
    resnet, classes = load_resnet_autodetect(MODEL_RESNET_PATH, USER_CLASSES, device=device_resnet)
    sanity_checks(resnet, classes, device_resnet)

    ensure_exists(args.image, "image")
    pil = Image.open(args.image).convert("RGB")

    out = analyze_image(
        pil_image=pil,
        yolo_model=yolo,
        resnet_model=resnet,
        classes=classes,
        device_resnet=device_resnet,
        conf=args.conf,
        iou=args.iou,
        topk=args.topk,
        return_drawn=True
    )

    # Impression console
    dets = out["results"]
    if len(dets) == 0:
        log("⚠️ Aucun avion détecté.")
    else:
        log("\n📋 Résultats:")
        for i, r in enumerate(dets, 1):
            x1, y1, x2, y2 = r["coordonnees"]
            log(f"\n🛩️ Avion {i}: bbox=({x1},{y1},{x2},{y2}) | conf={r['confiance_detection']:.3f}")
            for rank, cand in enumerate(r["topk"], 1):
                log(f"   #{rank} {cand['label']} — p={cand['prob']:.4f} (id={cand['class_id']})")

    # Sauvegarde image annotée si demandé
    if args.save:
        out_path = Path(args.save)
        out["annotated"].save(out_path)
        log(f"💾 Image annotée sauvegardée dans: {out_path}")

# ============================================================
# Gradio
# ============================================================
def build_ui():
    if not _HAS_GRADIO:
        raise RuntimeError("Gradio n'est pas installé. pip install gradio")

    device_yolo, device_resnet = get_devices()
    yolo = load_yolo(MODEL_YOLO_PATH)
    resnet, classes = load_resnet_autodetect(MODEL_RESNET_PATH, USER_CLASSES, device=device_resnet)
    sanity_checks(resnet, classes, device_resnet)

    def predict(pil_image: Optional[Image.Image], conf: float, iou: float, topk: int):
        if pil_image is None:
            return None, "Veuillez uploader une image.", None
        try:
            out = analyze_image(
                pil_image=pil_image,
                yolo_model=yolo,
                resnet_model=resnet,
                classes=classes,
                device_resnet=device_resnet,
                conf=conf,
                iou=iou,
                topk=topk,
                return_drawn=True
            )
            dets = out["results"]
            if len(dets) == 0:
                return out["annotated"], "Aucun avion détecté", None
            # Texte formaté
            lines = []
            for i, r in enumerate(dets, 1):
                x1, y1, x2, y2 = r["coordonnees"]
                lines.append(f"🛩️ Avion {i} - bbox=({x1},{y1},{x2},{y2}) | conf={r['confiance_detection']:.3f}")
                for rk, cand in enumerate(r["topk"], 1):
                    lines.append(f"   #{rk} {cand['label']} — p={cand['prob']:.4f} (id={cand['class_id']})")
            text = "\n".join(lines)
            main_crop = out["crops"][0] if len(out["crops"]) else None
            return out["annotated"], text, main_crop
        except Exception as e:
            return pil_image, f"❌ Exception: {type(e).__name__}: {e}", None

    with gr.Blocks(title="Détecteur d'avions + Top‑k") as demo:
        gr.Markdown("## ✈️ Détection d’avions et Top‑k classification")
        gr.Markdown("Uploader une image. Si un avion est détecté, on renvoie le Top‑k ResNet.")
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(type="pil", label="Image", image_mode="RGB")
                conf = gr.Slider(0.05, 0.9, v
                                 alue=YOLO_CONF_THRES, step=0.05, label="Seuil confiance YOLO")
                iou = gr.Slider(0.1, 0.9, value=YOLO_IOU_THRES, step=0.05, label="Seuil IoU YOLO")
                topk = gr.Slider(1, 5, value=3, step=1, label="Top‑k ResNet")
                btn = gr.Button("Analyser")
            with gr.Column():
                out_img = gr.Image(type="pil", label="Image annotée")
                out_txt = gr.Textbox(label="Résultats", lines=12)
                out_crop = gr.Image(type="pil", label="Crop du 1er avion (optionnel)")
        btn.click(fn=predict, inputs=[in_img, conf, iou, topk], outputs=[out_img, out_txt, out_crop])
        in_img.change(fn=predict, inputs=[in_img, conf, iou, topk], outputs=[out_img, out_txt, out_crop])
    return demo

# ============================================================
# Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Détection d'avions (YOLO) + classification (ResNet) Top‑k")
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