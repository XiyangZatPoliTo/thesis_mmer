# models/video_model_resnet_lstm.py
import torch
import torch.nn as nn
import torchvision.models as models

class VideoModelResNetLSTM(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 lstm_layers=1,
                 bidirectional=True,
                 freeze_backbone=True,
                 dropout=0.3):
        """
        Args:
            hidden_dim:  LSTM hidden dimension
            lstm_layers:  number of LSTM layers
            bidirectional:  use bidirectional LSTM if True
            freeze_backbone: if true, CNN weights will be frozen
            dropout: Dropout applied inside LSTM and before fusion
        """
        super(VideoModelResNetLSTM, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        cnn_modules = list(resnet.children())[:-1]  # 移除fc层
        self.backbone = nn.Sequential(*cnn_modules)  # 输出: [B, 512, 1, 1]

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.frame_feat_dim = 512

        self.use_lstm = True
        self.lstm = nn.LSTM(
            input_size=self.frame_feat_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)

    # -----------------------------------------------
    def forward(self, x):  # [B, T, 3, 224, 224]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.flatten(1)
        feats = feats.view(B, T, self.frame_feat_dim)  # [B, T, 512]

        lstm_out, _ = self.lstm(feats)
        video_emb = lstm_out[:, -1, :]
        return self.dropout(video_emb)  # [B, out_dim]
