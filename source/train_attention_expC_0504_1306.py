"""
多模态情绪识别训练脚本（ResNet18 + LSTM + Attention Fusion）
实验 C：更换视频模型结构（ViT → ResNet18）
训练时间：0504_1310
"""
import gc
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.video_model_resnet_lstm import VideoModelResNetLSTM
from models.attention_fusion import AttentionFusion
from utils.train_utils import EarlyStopping, save_training_logs, plot_training_curves
from utils.metrics import calculate_metrics

# ========== Config ==========
config = {
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "dropout": 0.3,
    "embedding_dim": 256,
    "num_classes": 8,
    "frames": 15,
    "patience": 6,
    "n_mels": 128,
    "grad_accum_steps": 1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ========== Dataset ==========
dataset = RAVDESSMultimodalDataset(
    dataset_dir="D:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24",
    mel_specs_kwargs={"n_mels": config["n_mels"]},
    frames=config["frames"],
    verbose=False
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"])

# ========== Models ==========
audio_model = AudioEmbedding(embedding_dim=config["embedding_dim"]).to(config["device"])
video_model = VideoModelResNetLSTM(hidden_dim=256, freeze_backbone=False).to(config["device"])
model = AttentionFusion(
    audio_dim=config["embedding_dim"],
    video_dim=video_model.out_dim,
    num_classes=config["num_classes"],
    dropout=config["dropout"]
).to(config["device"])

# ========== Optimizer & Loss ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
early_stopper = EarlyStopping(patience=config["patience"], verbose=True)

# ========== Training ==========
train_accs, val_accs, losses = [], [], []
print("🚀 开始训练（ResNet + Attention Fusion，实验 C）...")
start_time = time.time()

gc.collect()
torch.cuda.empty_cache()

for epoch in range(config["num_epochs"]):
    model.train()
    audio_model.train()
    video_model.train()
    epoch_start = time.time()
    total_loss, correct, total = 0.0, 0, 0

    for audio, video, labels in train_loader:
        audio, video, labels = audio.to(config["device"]), video.to(config["device"]), labels.to(config["device"])
        optimizer.zero_grad()
        audio_feat = audio_model(audio)
        video_feat = video_model(video)

        outputs = model(audio_feat, video_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    val_acc, _ = calculate_metrics(model, val_loader, config["device"], audio_model, video_model, mode="acc")

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    losses.append(total_loss / len(train_loader))

    duration = time.time() - epoch_start
    print(f"📘 Epoch {epoch+1}/{config['num_epochs']} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Time: {duration:.2f}s")

    early_stopper(val_acc)
    if early_stopper.early_stop:
        print("⛔ Early stopping triggered!")
        break

    torch.cuda.empty_cache()

# ========== Save ==========
version = config["version"]
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

torch.save(model.state_dict(), f"saved_models/resnet_attention_expC_{version}.pt")
save_training_logs(train_accs, val_accs, losses, f"logs/resnet_attention=_log_expB_{version}.csv")
plot_training_curves(train_accs, val_accs, losses, f"plots/resnet_attention_plot_expC_{version}.png")

total_time = time.time() - start_time
print(f"✅ 模型已保存至 saved_models/vit_attention_expB_{version}.pt")
print(f"✅ 日志已保存至 logs/vit_attention_log_expB_{version}.csv")
print(f"✅ 图表已保存至 plots/vit_attention_plot_expB_{version}.png")
print(f"⏱️ 总训练耗时：{total_time/60:.2f} 分钟")
