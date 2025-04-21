import torch
import torch.nn as nn
import torchvision.models as models

class VideoModel(nn.Module):
    def __init__(self, hidden_dim=256, lstm_layers=1, bidirectional=True, freeze_backbone=True):
        super(VideoModel, self).__init__()

        # Load pre-trained ResNet18
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove the final FC layer
        self.backbone = nn.Sequential(*modules)  # output: [B, 512, 1, 1]

        # Optionally freeze ResNet backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # LSTM to model temporal info across frames
        self.lstm_input_dim = 512
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, x):
        """
        x: [B, T, 3, 224, 224]  (batch of video frames)
        return: [B, out_dim]    (per-video embedding)
        """
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)  # Flatten time dimension

        # Extract per-frame features
        feats = self.backbone(x)          # [B*T, 512, 1, 1]
        feats = feats.view(B, T, -1)      # [B, T, 512]

        # Temporal modeling
        lstm_out, _ = self.lstm(feats)    # [B, T, H]
        video_embedding = lstm_out[:, -1, :]  # Use last step (or mean over T)

        return video_embedding
