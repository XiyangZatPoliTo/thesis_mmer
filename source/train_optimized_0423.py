
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import csv

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import get_audio_model
from models.video_model import VideoModel
from models.fusion_model import FusionModel


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        logprobs = self.log_softmax(x)
        true_dist = torch.zeros_like(logprobs)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))


class EarlyStopping:
    def __init__(self, patience=6, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_model(dataset_path, apply_augment=True, batch_size=2, accumulation_steps=4):
    dataset = RAVDESSMultimodalDataset(dataset_path, apply_augment=apply_augment)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    audio_model = get_audio_model(mode="embedding", embedding_dim=256, dropout=0.5)
    video_model = VideoModel(hidden_dim=256, dropout=0.5)
    fusion_model = FusionModel(audio_dim=256, video_dim=512, num_classes=8, dropout=0.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_model.to(device)
    video_model.to(device)
    fusion_model.to(device)

    criterion = LabelSmoothingLoss(classes=8, smoothing=0.1)
    optimizer = optim.Adam(
        list(audio_model.parameters()) + list(video_model.parameters()) + list(fusion_model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=6, verbose=True)

    log_file = f"train_optimized_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "DurationSeconds"])

    print(f"🚀 开始训练（优化版）...")

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

        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("⛔ 触发 EarlyStopping，提前终止训练。")
            break

    print(f"✅ 训练完成，日志已保存至: {log_file}")


if __name__ == "__main__":
    train_model("F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24", apply_augment=True)

