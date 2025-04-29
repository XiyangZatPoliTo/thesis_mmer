import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.audio_modelx import AudioEmbedding
from models.video_model_vit import VideoModelViT
from models.attention_fusion import AttentionFusion
from dataset.dataset import RAVDESSMultimodalDataset
from utils.train_utils import train_epoch, validate_epoch

import pandas as pd
from time import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === Dataset ===
dataset = RAVDESSMultimodalDataset(
    dataset_dir="F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24/",
    mel_specs_kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    frames=15,
    verbose=True
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

# === Models ===
audio_model = AudioEmbedding(embedding_dim=256)
video_model = VideoModelViT(hidden_dim=256, use_lstm=True)
model = AttentionFusion(audio_dim=256, video_dim=512, num_classes=8, hidden_dim=256, dropout=0.5)
audio_model.to(device)
video_model.to(device)
model.to(device)

# === Optimizer / Loss / Scheduler ===
params = list(audio_model.parameters()) + list(video_model.parameters()) + list(model.parameters())
optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# === Training Loop ===
epochs = 30
train_logs = []
best_acc = 0.0
early_stop_counter = 0
max_patience = 6

print("🚀 开始训练（ViT + Attention Fusion）...")
for epoch in range(1, epochs + 1):
    start = time()
    audio_model.train()
    video_model.train()
    model.train()
    train_loss, train_acc = train_epoch(audio_model, video_model, model, train_loader, optimizer, criterion, device)

    audio_model.eval()
    video_model.eval()
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = validate_epoch(audio_model, video_model, model, val_loader, criterion, device)

    end = time()
    log = {
        "epoch": epoch,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "time": round(end - start, 2)
    }
    train_logs.append(log)
    print(f"Epoch {epoch}/{epochs} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Time: {log['time']}s")

    # Early Stopping
    if val_acc > best_acc:
        best_acc = val_acc
        early_stop_counter = 0
        torch.save({
            'audio_model': audio_model.state_dict(),
            'video_model': video_model.state_dict(),
            'fusion_model': model.state_dict()
        }, f"fusion_vit_attention_best.pt")
    else:
        early_stop_counter += 1
        print(f"EarlyStopping counter: {early_stop_counter} / {max_patience}")
        if early_stop_counter >= max_patience:
            break

# === 保存日志 ===
log_path = "train_vit_attention_log.csv"
pd.DataFrame(train_logs).to_csv(log_path, index=False)
print(f"✅ 训练完成，日志已保存至: {log_path}")
