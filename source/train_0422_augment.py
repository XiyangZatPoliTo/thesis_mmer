
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import get_audio_model
from models.video_model import VideoModel
from models.fusion_model import FusionModel
from utils.metrics import compute_metrics

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_and_log(name, dataset, apply_augment):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                       generator=torch.Generator().manual_seed(42))
    # 在使用cuda时显存不够，降低batch_size:8->2
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

    audio_model = get_audio_model(mode="embedding", embedding_dim=256, dropout=0.3)
    video_model = VideoModel(hidden_dim=256, dropout=0.3)
    fusion_model = FusionModel(audio_dim=256, video_dim=512, num_classes=8, dropout=0.3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_model.to(device)
    video_model.to(device)
    fusion_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(audio_model.parameters()) + list(video_model.parameters()) + list(fusion_model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    log_file = f"trainlog_{name}.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "DurationSeconds"])

    print(f"🚀 开始训练 {'(增强版)' if apply_augment else '(原版)'}")

    for epoch in range(30):
        start_time = time.time()

        audio_model.train()
        video_model.train()
        fusion_model.train()

        train_loss, train_correct, total = 0, 0, 0
        for audio, video, labels in train_loader:
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            audio_feat = audio_model(audio)
            video_feat = video_model(video)
            logits = fusion_model(audio_feat, video_feat)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = train_correct / total

        audio_model.eval()
        video_model.eval()
        fusion_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for audio, video, labels in val_loader:
                audio, video, labels = audio.to(device), video.to(device), labels.to(device)
                audio_feat = audio_model(audio)
                video_feat = video_model(video)
                logits = fusion_model(audio_feat, video_feat)

                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        duration = time.time() - start_time

        print(f"[{name}] Epoch {epoch+1}/30 | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Time: {duration:.2f}s")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, duration])

    print(f"✅ {name} 完成，日志已保存至: {log_file}")
    return log_file


def plot_comparison(log1, log2, label1, label2):
    df1 = pd.read_csv(log1)
    df2 = pd.read_csv(log2)

    plt.figure()
    plt.plot(df1["Epoch"], df1["ValAcc"], label=f"{label1} ValAcc")
    plt.plot(df2["Epoch"], df2["ValAcc"], label=f"{label2} ValAcc")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_acc_comparison.png")

    plt.figure()
    plt.plot(df1["Epoch"], df1["TrainLoss"], label=f"{label1} TrainLoss")
    plt.plot(df2["Epoch"], df2["TrainLoss"], label=f"{label2} TrainLoss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_loss_comparison.png")


def train_model_with_accum(dataset_path, apply_augment=False, batch_size=2, accumulation_steps=4):
    dataset = RAVDESSMultimodalDataset(dataset_path, apply_augment=apply_augment)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    audio_model = get_audio_model(mode="embedding", embedding_dim=256, dropout=0.3)
    video_model = VideoModel(hidden_dim=256, dropout=0.3)
    fusion_model = FusionModel(audio_dim=256, video_dim=512, num_classes=8, dropout=0.3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_model.to(device)
    video_model.to(device)
    fusion_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(audio_model.parameters()) + list(video_model.parameters()) + list(fusion_model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    log_file = f"train_accum_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "DurationSeconds"])

    print(f"🚀 开始训练（使用梯度累积）...")

    for epoch in range(30):
        start_time = time.time()

        audio_model.train()
        video_model.train()
        fusion_model.train()

        train_loss, train_correct, total = 0, 0, 0
        optimizer.zero_grad()

        for i, (audio, video, labels) in enumerate(train_loader):
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            audio_feat = audio_model(audio)
            video_feat = video_model(video)
            logits = fusion_model(audio_feat, video_feat)

            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = train_correct / total

        audio_model.eval()
        video_model.eval()
        fusion_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for audio, video, labels in val_loader:
                audio, video, labels = audio.to(device), video.to(device), labels.to(device)
                audio_feat = audio_model(audio)
                video_feat = video_model(video)
                logits = fusion_model(audio_feat, video_feat)

                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        duration = time.time() - start_time

        print(f"Epoch {epoch+1}/30 | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Time: {duration:.2f}s")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, duration])

    print(f"✅ 训练完成，日志已保存至: {log_file}")

if __name__ == "__main__":
    # dataset_clean = RAVDESSMultimodalDataset("F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24", apply_augment=False)
    # dataset_aug = RAVDESSMultimodalDataset("F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24", apply_augment=True)

    # log_clean = train_and_log("baseline", dataset_clean, False)
    # log_aug = train_and_log("augmented", dataset_aug, True)

    # plot_comparison(log_clean, log_aug, "baseline", "augmented")

    train_model_with_accum("F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24", apply_augment=True)