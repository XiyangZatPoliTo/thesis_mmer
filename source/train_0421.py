"""
main updates:
optimizer, scheduler, validation, dropout, logging, plotting, save model with timestamps
"""

import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from timeit import default_timer as timer
from utils.print_time import print_train_time
import pandas as pd
import matplotlib.pyplot as plt

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import get_audio_model
from models.video_model import VideoModel
from models.fusion_model import FusionModel
from utils.metrics import compute_metrics, print_classification_report
from utils.plot import plot_training_curves

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

config = {
    "dataset_path": "F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24",
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-4,
    "audio_dim": 256,
    "video_dim": 512,
    "num_classes": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dropout": 0.3
}

dataset = RAVDESSMultimodalDataset(
    dataset_dir=config["dataset_path"],
    mel_specs_kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    frames=15,
    debug=False
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

audio_model = get_audio_model(mode="embedding", embedding_dim=config["audio_dim"], dropout=config["dropout"])
video_model = VideoModel(hidden_dim=config["video_dim"] // 2, dropout=config["dropout"])
fusion_model = FusionModel(config["audio_dim"], config["video_dim"], config["num_classes"], dropout=config["dropout"])

audio_model.to(config["device"])
video_model.to(config["device"])
fusion_model.to(config["device"])

criterion = nn.CrossEntropyLoss()
params = list(audio_model.parameters()) + list(video_model.parameters()) + list(fusion_model.parameters())
optimizer = optim.Adam(params, lr=config["lr"], weight_decay=1e-5)  # ✅ weight_decay added
# verbose can be removed, and use scheduler.get_last_lr() to monitor learning rates
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"training_log_{log_time}.csv"
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "StartTime", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "DurationSeconds"])

print("🚀 开始训练...")
train_start_time = timer()

for epoch in range(config["epochs"]):
    epoch_start = time.time()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    audio_model.train()
    video_model.train()
    fusion_model.train()

    train_loss, train_correct, train_total = 0, 0, 0
    for audio, video, labels in train_loader:
        audio, video, labels = audio.to(config["device"]), video.to(config["device"]), labels.to(config["device"])

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
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    audio_model.eval()
    video_model.eval()
    fusion_model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, video, labels in val_loader:
            audio, video, labels = audio.to(config["device"]), video.to(config["device"]), labels.to(config["device"])
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
    duration = time.time() - epoch_start

    print(f"📘 Epoch {epoch+1}/{config['epochs']} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {duration:.2f}s")

    val_metrics = compute_metrics(all_labels, all_preds)
    print(f"📊 Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
    # 可选：print_classification_report(all_labels, all_preds)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, time_str, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{duration:.2f}"])

train_end_time = timer()
print_train_time(train_start_time, train_end_time)

# 带时间戳的模型保存（含月日时）
time_tag = datetime.now().strftime("%m%d_%H")
model_filename = f"multimodal_model_{time_tag}.pt"
torch.save({
    "audio_model": audio_model.state_dict(),
    "video_model": video_model.state_dict(),
    "fusion_model": fusion_model.state_dict()
}, model_filename)

print(f"✅ 模型训练完毕，已保存为 {model_filename}")

# 📈 自动绘图
plot_training_curves(log_file, save_prefix=f"curve_{time_tag}")
