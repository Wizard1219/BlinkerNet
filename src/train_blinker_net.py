"""
Training script for BlinkerNet on the CARLA dataset.
Uses the pre-extracted JPEG frames produced by preprocess.py.
"""
import datetime
import logging
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from utils.blinker_net import BlinkerNet
from utils.utils import print_log
from utils.video_dataset import VideoDataset

DATA_PATH = os.environ.get("BLINKER_DATA", "data/dataset_carla")
MAX_LENGTH = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_video_sequence(batch):
    padded, labels, paths = [], [], []
    for video, label, path in batch:
        T = video.shape[1]
        if T < MAX_LENGTH:
            video = F.pad(video, (0, 0, 0, 0, 0, MAX_LENGTH - T))
        elif T > MAX_LENGTH:
            video = video[:, :MAX_LENGTH]
        padded.append(video)
        labels.append(label)
        paths.append(path)
    return torch.stack(padded), torch.tensor(labels), paths


def stratified_split(samples, seed=42, val_frac=0.15, test_frac=0.15):
    """Each class appears in all three splits."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for i, (_, cls) in enumerate(samples):
        by_class[cls].append(i)
    tr, va, te = [], [], []
    for cls in sorted(by_class):
        idxs = rng.permutation(by_class[cls]).tolist()
        n = len(idxs)
        n_va = max(1, round(val_frac * n))
        n_te = max(1, round(test_frac * n))
        n_tr = n - n_va - n_te
        tr += idxs[:n_tr]
        va += idxs[n_tr:n_tr + n_va]
        te += idxs[n_tr + n_va:]
    return tr, va, te


def train():
    run = wandb.init(
        project="2fa-flashing-blinkernet",
        config={
            "batch_size": 8,
            "epochs": 50,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "max_length": MAX_LENGTH,
            "use_frame_diff": True,
        },
    )
    config = run.config

    for k in ("train", "val", "test", "info"):
        run.define_metric(f"{k}/*", step_metric="info/epoch")

    os.makedirs("logs", exist_ok=True)
    log_dir = f"logs/blinkernet_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/train_output.log", filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO, force=True,
    )
    print_log(f"Log dir: {log_dir}")
    print_log(f"Device: {device}")
    print_log(f"Config: {dict(config)}")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = VideoDataset(root_dir=DATA_PATH, transform=transform)
    num_classes = len(dataset.classes)
    print_log(f"Dataset: {len(dataset)} videos, {num_classes} classes")

    tr_idx, va_idx, te_idx = stratified_split(
        [(s[0], s[1]) for s in dataset.samples], seed=42
    )
    tr = torch.utils.data.Subset(dataset, tr_idx)
    va = torch.utils.data.Subset(dataset, va_idx)
    te = torch.utils.data.Subset(dataset, te_idx)
    print_log(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")

    # Pre-extracted JPEG → safe to use multiple workers (no cv2)
    tr_loader = DataLoader(tr, batch_size=config["batch_size"], shuffle=True,
                           collate_fn=pad_video_sequence, num_workers=4)
    va_loader = DataLoader(va, batch_size=config["batch_size"], shuffle=False,
                           collate_fn=pad_video_sequence, num_workers=2)
    te_loader = DataLoader(te, batch_size=config["batch_size"], shuffle=False,
                           collate_fn=pad_video_sequence, num_workers=2)

    model = BlinkerNet(num_classes=num_classes,
                       use_frame_diff=config["use_frame_diff"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print_log(f"Model: BlinkerNet, params={n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["learning_rate"],
                            weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"])

    train_loss_h, train_acc_h = [], []
    val_loss_h, val_acc_h = [], []
    best_val_acc = 0.0

    for epoch in tqdm(range(config["epochs"])):
        run.log({"info/epoch": epoch + 1})

        # Train
        model.train()
        correct = total = 0
        cum_loss = 0.0
        for videos, labels, _ in tr_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(videos)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cum_loss += loss.item()
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            run.log({"train/batch_loss": loss.item()})

        train_acc = correct / total
        train_loss = cum_loss / len(tr_loader)
        train_acc_h.append(train_acc); train_loss_h.append(train_loss)
        run.log({"train/accuracy": train_acc * 100,
                 "train/avg_loss": train_loss})

        # Val
        model.eval()
        correct = total = 0; cum_loss = 0.0
        with torch.no_grad():
            for videos, labels, _ in va_loader:
                videos = videos.to(device); labels = labels.to(device)
                logits = model(videos)
                loss = criterion(logits, labels)
                cum_loss += loss.item()
                preds = logits.argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total
        val_loss = cum_loss / len(va_loader)
        val_acc_h.append(val_acc); val_loss_h.append(val_loss)
        run.log({"val/accuracy": val_acc * 100, "val/avg_loss": val_loss})

        print_log(
            f"Epoch {epoch+1:3d}/{config['epochs']}  "
            f"Train L={train_loss:.4f} A={train_acc*100:.1f}%  "
            f"Val L={val_loss:.4f} A={val_acc*100:.1f}%"
        )

        scheduler.step()
        run.log({"info/lr": scheduler.get_last_lr()[0]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{log_dir}/model_best.pt")

    print_log(f"Best val accuracy: {best_val_acc*100:.2f}%")
    print_log("Loading best checkpoint for test evaluation")
    model.load_state_dict(torch.load(f"{log_dir}/model_best.pt"))

    # Test
    model.eval()
    test_preds, test_labels_all, paths_all = [], [], []
    with torch.no_grad():
        correct = total = 0
        for videos, labels, paths in te_loader:
            videos = videos.to(device); labels = labels.to(device)
            logits = model(videos)
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            test_preds.append(preds); test_labels_all.append(labels)
            paths_all.extend(paths)
    test_acc = correct / total
    print_log(f"Test Accuracy: {test_acc*100:.2f}%")
    run.log({"test/accuracy": test_acc * 100})

    torch.save(model.state_dict(), f"{log_dir}/model_final.pt")

    all_preds = torch.cat(test_preds).cpu()
    all_labels = torch.cat(test_labels_all).cpu()

    # Plots
    sns.set_theme(context="paper", style="darkgrid", palette="colorblind",
                  rc={"lines.linewidth": 2})
    for metric, train_h, val_h, ylab, title in [
        ("loss", train_loss_h, val_loss_h, "Loss", "Loss"),
        ("accuracy",
         [a * 100 for a in train_acc_h], [a * 100 for a in val_acc_h],
         "Accuracy (%)", "Accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_h, label="Train")
        ax.plot(val_h, label="Val")
        ax.set(xlabel="Epoch", ylabel=ylab, title=title)
        ax.legend()
        fig.savefig(f"{log_dir}/{metric}_curve.png", bbox_inches="tight")
        plt.close(fig)

    # Confusion matrix
    cls_names = [c.split("_")[-1] for c in dataset.classes]
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=cls_names, yticklabels=cls_names)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix (test acc {test_acc*100:.1f}%)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(f"{log_dir}/confusion_matrix.png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
        labels=list(range(num_classes)), zero_division=0
    )
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print_log(f"Class {i:2d} ({cls_names[i]})  P={p:.2f}  R={r:.2f}  F1={f:.2f}")
    print_log(f"Macro  P={precision.mean():.2f}  R={recall.mean():.2f}  "
              f"F1={f1.mean():.2f}")

    np.savez(f"{log_dir}/metrics.npz",
             train_loss=train_loss_h, val_loss=val_loss_h,
             train_acc=train_acc_h, val_acc=val_acc_h,
             precision=precision, recall=recall, f1=f1)

    run.finish()


if __name__ == "__main__":
    train()
