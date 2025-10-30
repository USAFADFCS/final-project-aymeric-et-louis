#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import subprocess
import numpy as np
from collections import deque
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# === SUPPRESSION DES WARNINGS ===
warnings.filterwarnings("ignore", category=UserWarning)

# === CONFIGURATION ===
DATA_DIR = "/Users/a.cariven/Documents/USAFA/comp scien/final-project-aymeric-et-louis/demos/crop"
NUM_EPOCHS = 250
BATCH_SIZE = 64
NUM_WORKERS = 8
IMAGE_SIZE = 128
LEARNING_RATE = 0.01
MODEL_SAVE_PATH = "resnet_model_optimized.pth"

# === NETTOYAGE DU DATASET ===
def clean_dataset(data_dir):
    # Suppression des fichiers .DS_Store
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))
                print(f"✅ Suppression de {os.path.join(root, file)}")

    # Suppression des dossiers vides
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                shutil.rmtree(dir_path)
                print(f"⚠️ Suppression du dossier vide: {dir_path}")

# === VÉRIFICATION GPU (MPS) ===
def check_mps():
    try:
        # Vérification basique de MPS
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS non disponible")

        # Récupération des infos système (sans problème de guillemets)
        python_version = f"{torch.__version__}"
        macos_version = subprocess.getoutput("sw_vers -productVersion").strip()

        # Commande alternative pour éviter les guillemets
        gpu_info = subprocess.getoutput("system_profiler SPDisplaysDataType | grep 'Chipset Model' | head -1").strip()
        if not gpu_info:
            gpu_info = "Non détecté (peut-être un GPU intégré)"

        print(f"✅ Configuration validée:")
        print(f"   - PyTorch: {python_version}")
        print(f"   - macOS: {macos_version}")
        print(f"   - GPU: {gpu_info}")

        return torch.device("mps")

    except Exception as e:
        raise RuntimeError(
            f"❌ Erreur de configuration MPS: {str(e)}\n"
            "Solutions possibles:\n"
            "1. Mettez à jour macOS (Monterey 12.3+ requis)\n"
            "2. Installez PyTorch nightly: pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu\n"
            "3. Vérifiez que votre Mac a un GPU AMD/Apple (pas de NVIDIA)"
        )

# === DÉTECTION DES CLASSES ===
def get_num_classes(data_dir):
    clean_dataset(data_dir)  # Nettoyage avant détection
    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])
    if not class_names:
        raise ValueError(f"Aucune classe valide dans {data_dir}")
    print(class_names)
    return len(class_names), class_names

# === CHARGEMENT DES DONNÉES ===
def load_data(data_dir, image_size=128, batch_size=256):
    """
    Charge les données avec optimisations pour MPS.
    Args:
        data_dir (str): Chemin vers le dossier des images (format ImageFolder).
        image_size (int): Taille des images (carré). Default: 128.
        batch_size (int): Taille des batches. Default: 256 (optimal pour MPS).
    Returns:
        loader (DataLoader): DataLoader optimisé.
        dataset (ImageFolder): Dataset original.
        num_classes (int): Nombre de classes détectées.
    """
    # Transformation avec normalisation standard (ImageNet)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Redimensionnement forcé
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # CRITIQUE
    ])

    # Chargement du dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Vérification de l'équilibrage des classes
    labels = [y for _, y in dataset.samples]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"✅ Dataset chargé: {len(dataset)} images | {len(unique)} classes")
    print(f"   Répartition: {dict(zip(unique, counts))}")

    # Création du DataLoader optimisé pour MPS
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,               # 8 workers pour un Mac M1/M2
        pin_memory=True,           # OBLIGATOIRE pour MPS
        persistent_workers=True,    # Évite de recréer les workers
        prefetch_factor=2,          # Précharge 2 batches
        drop_last=True              # Évite les petits batches
    )

    return loader, dataset, len(unique)

# === MODÈLE OPTIMISÉ ===
def create_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Gel des couches (sauf la dernière)
    for param in model.parameters():
        param.requires_grad = False

    # Remplacement de la dernière couche
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# === BOUCLE D'ENTRAÎNEMENT ===
def train(model, loader, device, num_epochs=30, lr=LEARNING_RATE):
    """
    Boucle d'entraînement optimisée pour MPS.
    Args:
        model (nn.Module): Modèle ResNet.
        loader (DataLoader): DataLoader retourné par load_data().
        device (torch.device): 'mps' ou 'cpu'.
        num_epochs (int): Nombre d'époques. Default: 30.
        lr (float): Learning rate. Default: 0.0001 (optimal pour AdamW + MPS).
    Returns:
        history (dict): Historique des losses et temps d'entraînement.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.8, verbose=True)

    history = {
        "train_loss": [],
        "batch_times": [],
        "lr": []
    }

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_times = []
        progress_bar = tqdm(loader, desc=f"Époque {epoch+1}/{num_epochs}", leave=False)

        for inputs, labels in progress_bar:
            start_time = time.time()

            # Transfert vers le device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward + Backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Métriques
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            running_loss += loss.item()

            # Mise à jour de la barre de progression
            progress_bar.set_postfix(
                loss=f"{loss.item():.3f}",
                batch_time=f"{batch_time:.3f}s",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}"
            )

        # Métriques par époque
        avg_loss = running_loss / len(loader)
        avg_time = np.mean(batch_times)
        history["train_loss"].append(avg_loss)
        history["batch_times"].append(avg_time)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        # Ajustement du learning rate
        scheduler.step(avg_loss)

        print(f"✅ Époque {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Temps/batch: {avg_time:.3f}s (±{np.std(batch_times):.3f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return history


# === POINT D'ENTRÉE ===
if __name__ == "__main__":
    print("🚀 Initialisation...")
    device = check_mps()
    num_classes, class_names = get_num_classes(DATA_DIR)
    print(f"✅ Classes détectées ({num_classes}): {class_names[:5]}... (et {len(class_names)-5} autres)")

    loader, dataset, num_classes = load_data(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)

    print(f"Dataset chargé: {len(loader.dataset)} images ({len(loader)} batches)")

    model = create_model(num_classes, device)
    history = train(model, loader, device, NUM_EPOCHS)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Modèle sauvegardé sous {MODEL_SAVE_PATH}")

    print("\n=== RÉSUMÉ ===")
    print(f"Temps moyen par batch: {np.mean(history['batch_times']):.3f}s")
    print(f"Loss finale: {history['loss'][-1]:.4f}")
    print(f"Temps total d'entraînement: {np.sum(history['batch_times'])*NUM_EPOCHS/60:.1f} min")
