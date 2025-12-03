#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entraînement d'un classifieur d'avions militaires à partir de crops
(structure type ImageFolder).

- Split auto/validation stratifié
- Data augmentation pour le auto
- Fine-tuning partiel d'un ResNet18 pré-entraîné ImageNet
- Sauvegarde du meilleur modèle sur la validation
"""

import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

# =========================
# CONFIG GLOBALE
# =========================

# ⚠️ ADAPTER CE CHEMIN ⚠️
DATA_DIR = r"C:\Users\demeu\PycharmProjects\final-project-aymeric-et-louis\demos\crop"

MODEL_SAVE_PATH = "../../models/resnet/resnet18_military_aircraft_best.pth"

IMAGE_SIZE = 224           # plus adapté à ResNet qu'un 128x128
BATCH_SIZE = 64
NUM_EPOCHS = 60
VAL_RATIO = 0.2            # 20% du dataset pour la validation
LEARNING_RATE = 1e-3
NUM_WORKERS = 4            # passe à 0 si souci de multiprocessing
SEED = 42


# =========================
# UTILS
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        print("✅ Utilisation de CUDA (GPU NVIDIA)")
        return torch.device("cuda")
    else:
        print("⚠️ Aucun GPU détecté, utilisation du CPU")
        return torch.device("cpu")


def clean_dataset(data_dir: str):
    """
    Supprime les fichiers .DS_Store et les dossiers vides
    (utile si tu as manipulé le dataset sur macOS).
    """
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f == ".DS_Store":
                os.remove(os.path.join(root, f))

    for root, dirs, files in os.walk(data_dir, topdown=False):
        for d in dirs:
            p = os.path.join(root, d)
            if not os.listdir(p):
                os.rmdir(p)


# =========================
# DATASET WRAPPER GLOBAL
# =========================

class TransformSubset(Dataset):
    """
    Dataset wrapper permettant d'appliquer un transform différent
    à un sous-ensemble d'un ImageFolder.
    Compatible avec DataLoader multiprocess sur Windows.
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, target = self.dataset.samples[self.indices[idx]]
        img = self.dataset.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target


# =========================
# DATASETS & DATALOADERS
# =========================

def create_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
):
    """
    Crée les DataLoaders auto/val à partir d'un dossier ImageFolder.
    Split stratifié par classe.
    """

    # Transforms pour le auto (data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.15, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Transforms pour la validation (pas d'augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Dataset brut (sans transform)
    base_dataset = datasets.ImageFolder(root=data_dir, transform=None)

    targets = np.array(base_dataset.targets)
    num_classes = len(base_dataset.classes)

    print(f"✅ Dataset chargé : {len(base_dataset)} images, {num_classes} classes")
    print("   Classes :", base_dataset.classes)

    # --- split stratifié auto / val ---
    train_indices = []
    val_indices = []

    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * val_ratio)
        val_indices.extend(idx[:split])
        train_indices.extend(idx[split:])

    print(f"   → {len(train_indices)} images auto, {len(val_indices)} images val")

    # Création des datasets wrappés
    train_dataset = TransformSubset(base_dataset, train_indices, train_transform)
    val_dataset   = TransformSubset(base_dataset, val_indices,   val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # CPU only, donc False OK
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, base_dataset.classes


# =========================
# MODÈLE
# =========================

def create_model(num_classes: int, device: torch.device):
    """
    Crée un ResNet18 pré-entraîné, avec fine-tuning de layer4 + fc.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # On gèle d'abord tous les paramètres
    for param in model.parameters():
        param.requires_grad = False

    # On dé-gèle seulement layer4 + fc pour un fine-tuning léger
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


# =========================
# BOUCLES TRAIN / EVAL
# =========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Val  ", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


# =========================
# MAIN
# =========================

def main():
    set_seed(SEED)
    clean_dataset(DATA_DIR)
    device = get_device()

    train_loader, val_loader, class_names = create_dataloaders(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        VAL_RATIO,
        NUM_WORKERS,
    )

    num_classes = len(class_names)
    model = create_model(num_classes, device)

    # On n'optimise que les paramètres entraînables
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val_acc = 0.0
    best_epoch = -1

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Époque {epoch+1}/{NUM_EPOCHS} ===")

        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - start_time

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"📉 LR Reduced: {old_lr:.6f} → {new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Temps: {elapsed/60:.1f} min"
        )

        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Nouveau meilleur modèle sauvegardé (Val Acc: {best_val_acc*100:.2f}%)")

    print("\n=== RÉSUMÉ FINAL ===")
    print(f"Meilleure Val Acc: {best_val_acc*100:.2f}% (époque {best_epoch})")
    print(f"Modèle sauvegardé sous: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
