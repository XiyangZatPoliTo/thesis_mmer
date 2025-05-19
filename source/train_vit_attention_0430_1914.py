"""
多模态情绪识别训练脚本（ViT + Attention Fusion）
版本时间：0430_1914（自动生成）
设备自动检测，适配 GPU（推荐用于 RTX3060 以上）
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.video_model_vit import VideoModelViT
from models.attention_fusion import AttentionFusion
from utils.train_utils import EarlyStopping, save_training_logs, plot_training_curves
from utils.metrics import calculate_metrics
import time
from datetime import datetime

# ===== 配置参数 =====
config = {
    "batch_size": 8,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "dropout": 0.3,
    "embedding_dim": 256,
    "num_classes": 8,
    "frames": 8,
    "patience": 6,
    "max_len": 128,
    "n_mels": 128,
    "grad_accum_steps": 1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "version": datetime.now().strftime("%m%d_%H%M")
}

# ===== 数据准备 =====
dataset = RAVDESSMultimodalDataset(
    dataset_dir="D:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24",
    mel_specs_kwargs={"n_mels": config["n_mels"]},
    frames=config["frames"],
    verbose=False,
    debug=False,
    apply_augment=False,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"])

# ===== 模型定义 =====
audio_model = AudioEmbedding(embedding_dim=config["embedding_dim"]).to(config["device"])
video_model = VideoModelViT(hidden_dim=256, use_lstm=True).to(config["device"])
model = AttentionFusion(
    audio_dim=config["embedding_dim"],
    video_dim=video_model.out_dim,
    num_classes=config["num_classes"],
    dropout=config["dropout"]
).to(config["device"])

# ===== 优化器与损失函数 =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
early_stopper = EarlyStopping(patience=config["patience"], verbose=True)

# ===== 日志记录变量 =====
train_accs, val_accs, losses = [], [], []

print(f"🚀 开始训练（ViT + Attention Fusion v{config['version']}）...")

start_all = time.time()
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
        # print("Debug raw audio", audio.shape)
        audio_feat = audio_model(audio)
        # print("Debug raw video", video.shape)
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
    """
    # metrics用法
    # 仅返回准确率
    val_acc, _ = calculate_metrics(model, val_loader, config["device"], mode="acc")

    # 获取完整评估指标
    val_acc, metrics = calculate_metrics(model, val_loader, config["device"], mode="all")
    print(metrics["precision"], metrics["recall"], metrics["f1"])
    """

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    losses.append(total_loss / len(train_loader))

    elapsed = time.time() - epoch_start
    print(f"📘 Epoch {epoch+1}/{config['num_epochs']} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Time: {elapsed:.2f}s")

    early_stopper(val_acc)
    if early_stopper.early_stop:
        print("⛔ Early stopping triggered!")
        break

    torch.cuda.empty_cache()

# ===== 保存模型、日志、图表 =====
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

model_name = f"vit_attention_{config['version']}.pt"
log_path = f"logs/vit_attention_log_{config['version']}.csv"
plot_path = f"plots/vit_attention_plot_{config['version']}.png"
torch.save(model.state_dict(), f"saved_models/{model_name}")
save_training_logs(train_accs, val_accs, losses, log_path)
plot_training_curves(train_accs, val_accs, losses, plot_path)

print(f"✅ 模型已保存至: saved_models/{model_name}")
print(f"✅ 日志已保存至: {log_path}")
print(f"✅ 训练图已保存至: {plot_path}")
print(f"🏁 总训练时间: {(time.time() - start_all) / 60:.2f} 分钟")
