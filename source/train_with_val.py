from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import get_audio_model
from models.video_model import VideoModel
from models.fusion_model import FusionModel
from timeit import default_timer as timer
from utils.print_time import print_train_time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 配置 ===
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

# === 加载数据集并划分训练/验证 ===
full_dataset = RAVDESSMultimodalDataset(
    dataset_dir=config["dataset_path"],
    mel_specs_kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    frames=15,
    debug=False
)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# === 构建模型 ===
audio_model = get_audio_model(mode="embedding", embedding_dim=config["audio_dim"], dropout=config["dropout"])
video_model = VideoModel(hidden_dim=config["video_dim"] // 2, dropout=config["dropout"])
fusion_model = FusionModel(config["audio_dim"], config["video_dim"], config["num_classes"], dropout=config["dropout"])
audio_model.to(config["device"])
video_model.to(config["device"])
fusion_model.to(config["device"])

# === 优化器 & 损失函数 ===
criterion = nn.CrossEntropyLoss()
params = list(audio_model.parameters()) + list(video_model.parameters()) + list(fusion_model.parameters())
optimizer = optim.Adam(params, lr=config["lr"])

# === 训练记录 ===
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("🚀 开始训练...")
whole_train_start = timer()

for epoch in range(config["epochs"]):
    train_time_start = timer()
    audio_model.train()
    video_model.train()
    fusion_model.train()

    total_loss, correct, total = 0, 0, 0
    for audio, video, labels in train_loader:
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        labels = labels.to(config["device"])

        audio_feat = audio_model(audio)
        video_feat = video_model(video)
        logits = fusion_model(audio_feat, video_feat)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_losses.append(total_loss)
    train_accuracies.append(train_acc)

    # === 验证阶段 ===
    audio_model.eval()
    video_model.eval()
    fusion_model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for audio, video, labels in val_loader:
            audio = audio.to(config["device"])
            video = video.to(config["device"])
            labels = labels.to(config["device"])

            audio_feat = audio_model(audio)
            video_feat = video_model(video)
            logits = fusion_model(audio_feat, video_feat)

            loss = criterion(logits, labels)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"📘 Epoch {epoch+1}/{config['epochs']} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    train_time_end = timer()
    total_train_time = print_train_time(train_time_start, train_time_end)

whole_train_end = timer()
print("---------train end----------")
print_train_time(whole_train_start, whole_train_end)
# === 保存模型 ===
time_tag = datetime.now().strftime("%m%d_%H")
model_filename = f"multimodal_model_{time_tag}.pt"
torch.save({
    "audio_model": audio_model.state_dict(),
    "video_model": video_model.state_dict(),
    "fusion_model": fusion_model.state_dict()
}, model_filename)
print(f"✅ 模型训练完毕，已保存为 {model_filename}")

# === 绘制训练曲线 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# === 混淆矩阵 ===
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix (Validation Set)")
plt.savefig("confusion_matrix.png")
plt.show()
