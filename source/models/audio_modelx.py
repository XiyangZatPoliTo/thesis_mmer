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
# 增加input shape以解决mat1和mat2无法相乘的问题
# mat1 and mat2 shapes cannot be multiplied (1024x128 and 256x256)
class AudioEmbedding(nn.Module):
    def __init__(self, input_shape=(128, 128), embedding_dim=256, dropout=0.3):
        super(AudioEmbedding, self).__init__()
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim

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

        self.flatten_dim = 64 * (input_shape[0] // 8) * (input_shape[1] // 8)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim)
        )

        self.out_feature = embedding_dim

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 128, 128]
        x = self.conv(x)
        x = self.embedding(x)
        return x


# 简单工厂函数：支持训练脚本按需调用
def get_audio_model(mode="embedding", **kwargs):
    if mode == "classifier":
        return AudioClassifier(**kwargs)
    elif mode == "embedding":
        return AudioEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown audio model mode: {mode}")


# audio_model = AudioEmbedding(embedding_dim=256)
# print(audio_model(torch.randn(1, 128, 128)).shape)