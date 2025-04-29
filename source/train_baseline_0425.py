import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.video_model import VideoModel
from models.fusion_model import FusionModel
from utils.train_utils import EarlyStopping, AverageMeter, save_training_logs, plot_training_curves

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 固定随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()

# 配置参数
config = {
    "dataset_dir": "F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24/",
    "batch_size": 8,
    "frames": 10,
    "num_classes": 8,
    "audio_embedding_dim": 256,
    "video_embedding_dim": 512,
    "fusion_hidden_dim": 256,
    "dropout": 0.3,
    "epochs": 30,
    "patience": 6,
    "learning_rate": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "grad_accum_steps": 2,  # 梯度累积
}

# 创建数据集
dataset = RAVDESSMultimodalDataset(
    dataset_dir=config["dataset_dir"],
    mel_specs_kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    frames=config["frames"],
    verbose=True,
    debug=False,
    apply_augment=False
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=0)

# 创建模型
audio_model = AudioEmbedding(embedding_dim=config["audio_embedding_dim"]).to(config["device"])
video_model = VideoModel(hidden_dim=config["video_embedding_dim"] // 2, lstm_layers=1, bidirectional=True,
                         freeze_backbone=True).to(config["device"])
model = FusionModel(
    audio_dim=config["audio_embedding_dim"],
    video_dim=config["video_embedding_dim"],
    num_classes=config["num_classes"],
    hidden_dim=config["fusion_hidden_dim"],
    dropout=config["dropout"]
)

model = model.to(config["device"])

# 优化器、损失函数、学习率调度器
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# EarlyStopping
early_stopper = EarlyStopping(patience=config["patience"], verbose=True)

# 保存日志
log_list = []

print("🚀 开始训练（Baseline）...")

for epoch in range(1, config["epochs"] + 1):
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    optimizer.zero_grad()

    for step, (audio, video, label) in enumerate(train_loader):
        audio = audio.to(config["device"])
        video = video.to(config["device"])
        label = label.to(config["device"])

        audio_feat = audio_model(audio)
        video_feat = video_model(video)
        output = model(audio_feat, video_feat)

        loss = criterion(output, label)
        loss = loss / config["grad_accum_steps"]
        loss.backward()

        # 累积更新
        if (step + 1) % config["grad_accum_steps"] == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        preds = torch.argmax(output, dim=1)
        acc = (preds == label).float().mean()

        train_loss_meter.update(loss.item() * config["grad_accum_steps"], n=audio.size(0))
        train_acc_meter.update(acc.item(), n=audio.size(0))

    # 验证
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    with torch.no_grad():
        for audio, video, label in val_loader:
            audio = audio.to(config["device"])
            video = video.to(config["device"])
            label = label.to(config["device"])

            audio_feat = audio_model(audio)
            video_feat = video_model(video)
            output = model(audio_feat, video_feat)

            loss = criterion(output, label)
            preds = torch.argmax(output, dim=1)
            acc = (preds == label).float().mean()

            val_loss_meter.update(loss.item(), n=audio.size(0))
            val_acc_meter.update(acc.item(), n=audio.size(0))

    print(f"Epoch {epoch}/{config['epochs']} | TrainAcc: {train_acc_meter.avg:.4f} | ValAcc: {val_acc_meter.avg:.4f}")

    log_list.append({
        "epoch": epoch,
        "train_acc": train_acc_meter.avg,
        "val_acc": val_acc_meter.avg,
        "train_loss": train_loss_meter.avg,
        "val_loss": val_loss_meter.avg,
    })

    scheduler.step(val_loss_meter.avg)

    early_stopper(val_loss_meter.avg)
    if early_stopper.early_stop:
        print("⏹️ Early stopping triggered!")
        break

# 保存模型
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "fusion_baseline.pt")
torch.save(model.state_dict(), model_path)
print(f"✅ 模型已保存至: {model_path}")

# 保存训练日志
save_training_logs(log_list, prefix="baseline")
plot_training_curves(log_list, prefix="baseline")

print("🎉 Baseline训练流程完成！")
