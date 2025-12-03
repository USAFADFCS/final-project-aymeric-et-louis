#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# CONFIG
# =========================
DATA_DIR = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis/demos/crop"  # doit contenir auto/ et val/ ou un seul dossier (split auto)
EPOCHS = 5
BATCH_SIZE = 64
NUM_WORKERS = 8
IMAGE_SIZE = 128
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15            # utilisé seulement si pas de dossier val/
EARLY_STOP_PATIENCE = 10    # patience en époques, sur la val_loss
GRAD_CLIP_NORM = 1.0

# Scheduler plateau par itération
PLATEAU_FACTOR = 0.5        # multiplicateur LR quand on réduit
PLATEAU_PATIENCE = 3        # itérations consécutives sans amélioration
PLATEAU_THRESHOLD = 1e-4    # amélioration relative minimale
PLATEAU_MIN_LR = 1e-7
PLATEAU_COOLDOWN = 0        # itérations après réduction durant lesquelles on n’applique plus de réduction

# Lissage EMA optionnel de la loss (pour scheduler)
USE_EMA = True
EMA_ALPHA = 0.1             # plus petit = plus lisse

# Checkpoints (state_dict légers)
CKPT_DIR = "./checkpoints"
CKPT_BEST = os.path.join(CKPT_DIR, "../../models/yolo/ckpt_best_light.pth")
CKPT_LAST = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis/resnet_model_optimized.pth"
RESUME_FROM = CKPT_LAST 

SEED = 42

# =========================
# UTILS
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def imagenet_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def build_datasets(data_dir: str, img_size: int, val_split: float) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, List[str]]:
    train_tf, val_tf = imagenet_transforms(img_size)

    # Cas 1: data_dir contient "auto" et "val"
    train_dir = os.path.join(data_dir, "auto")
    val_dir = os.path.join(data_dir, "val")
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        classes = train_ds.classes
        return train_ds, val_ds, classes

    # Cas 2: split automatique depuis un seul dossier
    full_ds = datasets.ImageFolder(data_dir, transform=train_tf)
    n_total = len(full_ds)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    # Important: val doit utiliser val_tf (remplace transform du sous-dataset)
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_tf)
    classes = full_ds.classes
    return train_ds, val_ds, classes

def build_dataloaders(train_ds, val_ds, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_dl, val_dl

def build_model(num_classes: int) -> nn.Module:
    # ResNet18 pré-entraînée ImageNet; on remplace la tête
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        # fallback pour versions plus anciennes
        model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def save_light(path: str, model: nn.Module, optimizer: optim.Optimizer,
               epoch: int, best_val_loss: float, best_val_acc: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "models": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc
    }, path)

def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, dict): 
        return False
    # heuristique simple: au moins une clé de poids
    return any(isinstance(k, str) and (k.endswith(".weight") or ".running_mean" in k or ".bias" in k) 
               for k in obj.keys())

def load_light(path: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None,
               map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)

    # 1) Cas: ckpt est directement un state_dict
    if _looks_like_state_dict(ckpt):
        model.load_state_dict(ckpt, strict=False)
        if optimizer is not None:
            # pas d’info optimizer disponible
            pass
        return {"epoch": 0, "best_val_loss": float("inf"), "best_val_acc": 0.0}

    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint inattendu: type={type(ckpt)}")

    # 2) Essayer plusieurs clés possibles pour le modèle
    model_sd = None
    for k in ["models", "state_dict", "model_state_dict", "net", "module", "params"]:
        if k in ckpt and _looks_like_state_dict(ckpt[k]):
            model_sd = ckpt[k]
            break
    if model_sd is None:
        # dernier recours: peut-être ckpt est un dict qui contient DIRECTEMENT les poids (ex: lightning avec prefix)
        # on filtre les clés plausibles
        candidate = {k: v for k, v in ckpt.items() if isinstance(k, str) and _looks_like_state_dict({k: v})}
        if candidate:
            model_sd = candidate
    if model_sd is None:
        raise KeyError(f"Aucune clé de state_dict trouvée dans {path}. Clés disponibles: {list(ckpt.keys())[:20]}")

    # Charger tolérant
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        print(f"⚠️ load_state_dict: missing keys (extrait): {missing[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"⚠️ load_state_dict: unexpected keys (extrait): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    # 3) Optimizer éventuel
    if optimizer is not None:
        opt_sd = None
        for k in ["optimizer", "optimizer_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                opt_sd = ckpt[k]
                break
        if opt_sd is not None:
            try:
                optimizer.load_state_dict(opt_sd)
            except Exception as e:
                print(f"⚠️ Impossible de charger l’optimizer (on repart neuf): {e}")

    # 4) Métadonnées
    meta = {
        "epoch": ckpt.get("epoch", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "best_val_acc": ckpt.get("best_val_acc", 0.0)
    }
    return meta

# =========================
# Scheduler plateau par itération
# =========================
class IterPlateau:
    def __init__(self, optimizer, factor=0.5, patience=3,
                 threshold=1e-4, threshold_mode="rel",
                 min_lr=1e-7, cooldown=0, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode  # "rel" ou "abs"
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.best = None
        self.num_bad = 0
        self.verbose = verbose

    def _is_better(self, loss, best):
        if best is None:
            return True
        if self.threshold_mode == "rel":
            return loss < best * (1 - self.threshold)
        else:
            return loss < best - self.threshold

    def _reduce_lr(self):
        changed = []
        for i, pg in enumerate(self.optimizer.param_groups):
            old_lr = float(pg["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr - 1e-16:
                pg["lr"] = new_lr
                changed.append((i, old_lr, new_lr))
        if self.verbose and changed:
            msg = " | ".join([f"group {i}: {old:.2e} -> {new:.2e}" for i, old, new in changed])
            print(f"⚠️ Reduce LR on plateau: {msg}")

    def step(self, loss_value: float):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if self._is_better(loss_value, self.best):
            self.best = loss_value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience and self.cooldown_counter == 0:
                self._reduce_lr()
                self.num_bad = 0
                self.cooldown_counter = self.cooldown

# =========================
# EVAL
# =========================
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    loss_sum = 0.0
    total = 0
    top1_sum = 0.0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            top1 = accuracy(logits, y, topk=(1,))[0]
            top1_sum += top1 * y.size(0) / 100.0
            total += y.size(0)
    return loss_sum / total, 100.0 * (top1_sum / total)

# =========================
# TRAIN
# =========================
def train():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_ds, val_ds, classes = build_datasets(DATA_DIR, IMAGE_SIZE, VAL_SPLIT)
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    train_dl, val_dl = build_dataloaders(train_ds, val_ds, BATCH_SIZE, NUM_WORKERS)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    # Scheduler plateau par itération
    plateau = IterPlateau(
        optimizer,
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=PLATEAU_THRESHOLD,
        threshold_mode="rel",
        min_lr=PLATEAU_MIN_LR,
        cooldown=PLATEAU_COOLDOWN,
        verbose=True
    )

    # Reprise éventuelle
start_epoch = 0
best_val_loss = float("inf")
best_val_acc = 0.0
if RESUME_FROM and os.path.isfile(RESUME_FROM):
    meta = load_light(RESUME_FROM, models, optimizer, map_location="cpu")
    start_epoch = int(meta.get("epoch", 0)) + 1
    best_val_loss = float(meta.get("best_val_loss", float("inf")))
    best_val_acc = float(meta.get("best_val_acc", 0.0))
    print(f"Reprise depuis {RESUME_FROM} à l’époque {start_epoch}.")

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}
    no_improve_epochs = 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        ema = None

        pbar = tqdm(train_dl, desc=f"Époque {epoch+1}/{EPOCHS}")
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            optimizer.step()

            # Scheduler plateau par itération (sur loss brute ou EMA)
            loss_val = loss.item()
            if USE_EMA:
                ema = loss_val if ema is None else EMA_ALPHA * loss_val + (1 - EMA_ALPHA) * ema
                plateau.step(float(ema))
            else:
                plateau.step(loss_val)

            running_loss += loss_val
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{cur_lr:.2e}")

        train_loss = running_loss / max(1, len(train_dl))
        val_loss, val_acc = evaluate(model, val_dl, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"✅ Époque {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%")

        # Sauvegarde “best” si amélioration val_loss
        improved = val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_loss
            best_val_acc = max(best_val_acc, val_acc)
            save_light(CKPT_BEST, model, optimizer, epoch, best_val_loss, best_val_acc)
            print(f"💾 Nouveau meilleur modèle → {CKPT_BEST} | val_loss={best_val_loss:.4f} | best_acc={best_val_acc:.2f}%")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping sur val_loss
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"⏹️ Early stopping (patience={EARLY_STOP_PATIENCE})")
            break

    # Sauvegarde “last”
    last_epoch = start_epoch + len(history["train_loss"]) - 1
    save_light(CKPT_LAST, model, optimizer, last_epoch, best_val_loss, best_val_acc)
    print(f"💾 Checkpoint final (last) → {CKPT_LAST}")

    # Résumé
    print("\n=== RÉSUMÉ ===")
    print(f"Meilleure val_loss: {best_val_loss:.4f} | Meilleure val_acc: {best_val_acc:.2f}%")
    print(f"Dernière train_loss: {history['train_loss'][-1]:.4f}")
    print(f"Dernière val_loss:   {history['val_loss'][-1]:.4f} | val_acc: {history['val_acc'][-1]:.2f}%")

if __name__ == "__main__":
    train()
