import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import get_audio_model
from models.video_model import VideoModel
from models.fusion_model import FusionModel
import os
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
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# === 数据集加载 ===
dataset = RAVDESSMultimodalDataset(
    dataset_dir=config["dataset_path"],
    mel_specs_kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    frames=15,
    debug=False
)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# === 构建模型 ===
audio_model = get_audio_model(mode="embedding", embedding_dim=config["audio_dim"])
video_model = VideoModel(hidden_dim=config["video_dim"] // 2)  # 双向LSTM输出翻倍
fusion_model = FusionModel(config["audio_dim"], config["video_dim"], config["num_classes"])

# 将模型移至设备
audio_model.to(config["device"])
video_model.to(config["device"])
fusion_model.to(config["device"])

# === 训练组件 ===
criterion = nn.CrossEntropyLoss()
params = list(audio_model.parameters()) + list(video_model.parameters()) + list(fusion_model.parameters())
optimizer = optim.Adam(params, lr=config["lr"])

# === 训练循环 ===
print("🚀 开始训练...")
for epoch in range(config["epochs"]):
    audio_model.train()
    video_model.train()
    fusion_model.train()

    total_loss, correct, total = 0, 0, 0

    for audio, video, labels in dataloader:
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        labels = labels.to(config["device"])

        # 前向传播
        audio_feat = audio_model(audio)           # [B, D1]
        video_feat = video_model(video)           # [B, D2]
        logits = fusion_model(audio_feat, video_feat)  # [B, num_classes]

        # 计算损失与优化
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计信息
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"📘 Epoch {epoch+1}/{config['epochs']} | Loss: {total_loss:.4f} | Accuracy: {acc:.4f}")

# === 保存模型 ===
torch.save({
    "audio_model": audio_model.state_dict(),
    "video_model": video_model.state_dict(),
    "fusion_model": fusion_model.state_dict()
}, "multimodal_model_0419_1.pt")

print("✅ 模型训练完毕，已保存为 multimodal_model_0419_1.pt")
