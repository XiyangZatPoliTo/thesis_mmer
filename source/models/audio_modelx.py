# 实验性质的audio model
import torch
import torch.nn as nn
import torch.nn.functional as F


# 原版 EmoCatcher 分类模型：用于单模态情绪分类
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=8, dropout=0.3):
        super(AudioClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 128, 128]
        x = self.conv(x)
        return self.classifier(x)


# 多模态融合使用：输出音频嵌入特征而不是直接分类
class AudioEmbedding(nn.Module):
    def __init__(self, embedding_dim=256, dropout=0.3):
        super(AudioEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 128, 128]
        x = self.conv(x)
        return self.embedding(x)


# 简单工厂函数：支持训练脚本按需调用
def get_audio_model(mode="embedding", **kwargs):
    if mode == "classifier":
        return AudioClassifier(**kwargs)
    elif mode == "embedding":
        return AudioEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown audio model mode: {mode}")
