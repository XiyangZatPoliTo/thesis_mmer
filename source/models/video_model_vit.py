import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class VideoModelViT(nn.Module):
    """
    ViT-based VideoModel：将视频帧序列输入 ViT，提取特征后融合输出。
    可选是否使用 LSTM 对帧序列进行建模。
    """
    def __init__(self, hidden_dim=256, pretrained=True, use_lstm=False, lstm_layers=1, bidirectional=True):
        super(VideoModelViT, self).__init__()
        self.use_lstm = use_lstm
        self.vit = vit_b_16(pretrained=pretrained)
        self.vit.heads = nn.Identity()  # 移除分类头部
        self.frame_feat_dim = self.vit.hidden_dim  # 通常为768

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.frame_feat_dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
            self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        else:
            self.out_dim = self.frame_feat_dim

    def forward(self, x):
        """
        输入：
            x: [B, T, 3, 224, 224]，表示视频帧序列
        输出：
            每个视频的聚合特征向量: [B, out_dim]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)  # [B*T, 768]
        feats = feats.view(B, T, -1)  # [B, T, 768]

        if self.use_lstm:
            lstm_out, _ = self.lstm(feats)
            return lstm_out[:, -1, :]
        else:
            return feats.mean(dim=1)
