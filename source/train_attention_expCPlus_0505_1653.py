"""
多模态情绪识别训练脚本（ResNet18 + LSTM + Attention Fusion）
实验 C：更换视频模型结构（ViT → ResNet18）
训练时间：0504_1310
"""
import gc
import math
import os
import time
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.amp import autocast, GradScaler
scaler = GradScaler(device='cuda')

from torch.utils.data import DataLoader, random_split
from dataset.dataset import RAVDESSMultimodalDataset
from models.audio_modelx import AudioEmbedding
from models.video_model_resnet_lstm import VideoModelResNetLSTM
from models.attention_fusion import AttentionFusion
from utils.train_utils import EarlyStopping, save_training_logs, plot_training_curves
from utils.metrics import calculate_metrics


# ========== Config ==========
config = {
    "exp_name": "train_attention_ExpC_Plus",
    "batch_size": 24,
    "num_epochs": 40,
    # "learning_rate": 1e-4,
    "lr_init": 3e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "embedding_dim": 256,
    "num_classes": 8,
    "frames": 15,
    "patience": 10,
    "n_mels": 128,
    "grad_accum_steps": 2,  # set-1 if fits RAM
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

# ========== data augmentation ==========
freq_mask = T.FrequencyMasking(freq_mask_param=16)
time_mask = T.TimeMasking(time_mask_param=24)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_init"], weight_decay=config["weight_decay"])
step_per_epoch = math.ceil(len(train_loader) / config["grad_accum_steps"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config["num_epochs"], eta_min=config["min_lr"])
scaler = GradScaler(device=config["device"])
early_stopper = EarlyStopping(patience=config["patience"], verbose=True)

# ========== Training ==========
train_accs, val_accs, losses = [], [], []
global_step = 0
print(f"🚀 开始训练 [{config['exp_name']}] on {config['device']} ...")
start_time = time.time()

gc.collect()
torch.cuda.empty_cache()

for epoch in range(config["num_epochs"]):
    model.train()
    audio_model.train()
    video_model.train()
    epoch_start = time.time()
    epoch_loss, correct, total = 0.0, 0, 0

    for step, (audio, video, labels) in enumerate(train_loader):
        audio, video, labels = audio.to(config["device"]), video.to(config["device"]), labels.to(config["device"])


        # ------ augmentation ---------
        audio = freq_mask(time_mask(audio))
        # print(video.shape)
        if torch.rand(1) < 0.5:  # video flip
            video = torch.flip(video, dims=[4])  # horizontal

        with autocast(device_type='cuda'):
            audio_feat = audio_model(audio)
            video_feat = video_model(video)
            output = model(audio_feat, video_feat)
            loss = criterion(output, labels) / config["grad_accum_steps"]

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * config["grad_accum_steps"]
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step+1) % config["grad_accum_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        global_step += 1

    train_acc = correct / total
    val_acc, _ = calculate_metrics(model, val_loader, config["device"], audio_model, video_model, mode="acc")

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    losses.append(epoch_loss / len(train_loader))

    duration = time.time() - epoch_start
    print(f"📘 Epoch {epoch+1}/{config['num_epochs']} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | Time: {duration/60:.2f}min")

    early_stopper(val_acc)
    if early_stopper.early_stop:
        print("⛔ Early stopping triggered!")
        break

    torch.cuda.empty_cache()

# ========== Save ==========
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

stamp = time.strftime("%Y%m%d-%H%M%S")
model_path = f"saved_models/{config['exp_name']}_{stamp}.pt"
log_path = f"logs/{config['exp_name']}_{stamp}.csv"
plot_path = f"plots/{config['exp_name']}_{stamp}.png"


torch.save(model.state_dict(), model_path)
save_training_logs(train_accs, val_accs, losses, log_path)
plot_training_curves(train_accs, val_accs, losses, plot_path)

total_time = time.time() - start_time
print(f"✅ 模型已保存至 {model_path}")
print(f"✅ 日志已保存至 {log_path}")
print(f"✅ 图表已保存至 {plot_path}")
print(f"⏱️ 总训练耗时：{total_time/60:.2f} 分钟")
