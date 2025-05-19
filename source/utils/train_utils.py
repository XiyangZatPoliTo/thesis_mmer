import torch
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import csv

def train_epoch(audio_model, video_model, fusion_model, loader, optimizer, criterion, device, grad_accum=1):
    audio_model.train()
    video_model.train()
    fusion_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, (audio, video, label) in enumerate(loader):
        audio = audio.to(device)  # [B, 128, 128]
        video = video.to(device)  # [B, T, 3, 224, 224]
        label = label.to(device)

        audio_feat = audio_model(audio)
        video_feat = video_model(video)
        logits = fusion_model(audio_feat, video_feat)

        loss = criterion(logits, label) / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum
        pred = torch.argmax(logits, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    acc = correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc


def validate_epoch(audio_model, video_model, fusion_model, loader, criterion, device):
    audio_model.eval()
    video_model.eval()
    fusion_model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for audio, video, label in loader:
            audio = audio.to(device)
            video = video.to(device)
            label = label.to(device)

            audio_feat = audio_model(audio)
            video_feat = video_model(video)
            logits = fusion_model(audio_feat, video_feat)

            loss = criterion(logits, label)
            running_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    acc = correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=6, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def save_training_logs(log_list, prefix="baseline"):
#     """Save training and validation logs to CSV."""
#     os.makedirs("logs", exist_ok=True)
#     df = pd.DataFrame(log_list)
#     timestamp = time.strftime('%m%d_%H%M')
#     save_path = os.path.join("logs", f"{prefix}_train_log_{timestamp}.csv")
#     df.to_csv(save_path, index=False)
#     print(f"✅ 训练日志已保存至: {save_path}")

def save_training_logs(train_accs, val_accs, losses, filepath):
    """
    保存训练日志为 CSV 文件，包括每轮的训练准确率、验证准确率、训练损失。
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Accuracy", "Val Accuracy", "Loss"])
        for epoch, (train_acc, val_acc, loss) in enumerate(zip(train_accs, val_accs, losses), 1):
            writer.writerow([epoch, f"{train_acc:.4f}", f"{val_acc:.4f}", f"{loss:.4f}"])

# def plot_training_curves(log_list, prefix="baseline"):
#     """Plot and save training loss/accuracy curves."""
#     os.makedirs("plots", exist_ok=True)
#     df = pd.DataFrame(log_list)
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#     # 绘制准确率
#     ax1.plot(df["epoch"], df["train_acc"], label="Train Acc")
#     ax1.plot(df["epoch"], df["val_acc"], label="Val Acc")
#     ax1.set_title("Accuracy Curve")
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Accuracy")
#     ax1.legend()
#
#     # 绘制loss
#     ax2.plot(df["epoch"], df["train_loss"], label="Train Loss")
#     ax2.plot(df["epoch"], df["val_loss"], label="Val Loss")
#     ax2.set_title("Loss Curve")
#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("Loss")
#     ax2.legend()
#
#     plt.tight_layout()
#     timestamp = time.strftime('%m%d_%H%M')
#     save_path = os.path.join("plots", f"{prefix}_training_curves_{timestamp}.png")
#     fig.savefig(save_path)
#     print(f"✅ 训练曲线已保存至: {save_path}")
#     plt.close(fig)

def plot_training_curves(train_accs, val_accs, losses, save_path):
    """
    绘制训练准确率、验证准确率和损失曲线，并保存为 PNG。
    """
    if not (len(train_accs) == len(val_accs) == len(losses)):
        raise ValueError("Input lists must have the same length.")

    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(12, 6))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, label='Training Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to: {save_path}")