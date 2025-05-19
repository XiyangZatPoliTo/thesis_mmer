import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import time
from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.video_model_vit import VideoModelViT
from models.attention_fusion import AttentionFusion
from utils.train_utils import AverageMeter, EarlyStopping, save_training_logs, plot_training_curves
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===== 配置 =====
# RTX3060
config = {
    "dataset_dir": "D:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24/",
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "frames": 15,
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 6,
    "dropout": 0.3,
    "num_classes": 8,
    "embedding_dim": 256,
    "video_hidden_dim": 256,
    "use_lstm": True,
    "grad_accum_steps": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ===== 加载数据 =====
dataset = RAVDESSMultimodalDataset(
    dataset_dir=config["dataset_dir"],
    mel_specs_kwargs={"n_mels": config["n_mels"], "n_fft": config["n_fft"], "hop_length": config["hop_length"]},
    frames=config["frames"],
    verbose=True,
    debug=False,
    apply_augment=False
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

# ===== 创建模型 =====
audio_model = AudioEmbedding(embedding_dim=config["embedding_dim"]).to(config["device"])
video_model = VideoModelViT(hidden_dim=config["video_hidden_dim"], use_lstm=config["use_lstm"]).to(config["device"])

model = AttentionFusion(
    audio_dim=audio_model.embedding[-1].out_features,
    video_dim=video_model.out_dim,
    num_classes=config["num_classes"],
    dropout=config["dropout"]
).to(config["device"])

# ===== 损失函数 & 优化器 =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# ===== EarlyStopping & 日志器 =====
early_stopper = EarlyStopping(patience=config["patience"])
train_acc_meter = AverageMeter()
val_acc_meter = AverageMeter()

# ===== 开始训练 =====
print("🚀 开始训练（Attention + ViT，小训练版）...")
start_time = time.time()

train_acc_list, val_acc_list = [], []

for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for idx, (audio, video, label) in enumerate(train_loader):
        audio, video, label = audio.to(config["device"]), video.to(config["device"]), label.to(config["device"])

        # 分别提取特征
        audio_feat = audio_model(audio)
        video_feat = video_model(video)

        # 融合并分类
        output = model(audio_feat, video_feat)
        loss = criterion(output, label)
        loss = loss / config["grad_accum_steps"]
        loss.backward()

        if (idx + 1) % config["grad_accum_steps"] == 0 or (idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config["grad_accum_steps"]
        _, preds = output.max(1)
        correct += preds.eq(label).sum().item()
        total += label.size(0)

    train_acc = correct / total
    train_acc_list.append(train_acc)

    # ====== 验证集 ======
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for audio, video, label in val_loader:
            audio, video, label = audio.to(config["device"]), video.to(config["device"]), label.to(config["device"])

            audio_feat = audio_model(audio)
            video_feat = video_model(video)
            output = model(audio_feat, video_feat)

            _, preds = output.max(1)
            val_correct += preds.eq(label).sum().item()
            val_total += label.size(0)

    val_acc = val_correct / val_total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{config['epochs']} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f}")

    early_stopper(val_acc)
    if early_stopper.early_stop:
        print("⏸ 触发EarlyStopping，提前停止训练")
        break

# ===== 保存模型、日志、曲线 =====
save_time = time.strftime("%m%d_%H%M")
os.makedirs("saved_models", exist_ok=True)

model_save_path = f"saved_models/fusion_attention_vit_{save_time}.pt"
torch.save(model.state_dict(), model_save_path)
print(f"✅ 模型已保存至: {model_save_path}")

save_training_logs(train_acc_list, val_acc_list)
plot_training_curves(train_acc_list, val_acc_list)
print(f"✅ 训练日志已保存")
print(f"✅ 训练曲线已保存")
print(f"🎉 训练完成，耗时: {time.time() - start_time:.2f}s")
