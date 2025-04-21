import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, audio_dim, video_dim, num_classes=8, hidden_dim=256, dropout=0.3):
        super(FusionModel, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio_feat, video_feat):
        """
        audio_feat: [B, D1]
        video_feat: [B, D2]
        returns logits: [B, num_classes]
        """
        x = torch.cat([audio_feat, video_feat], dim=1)
        return self.fusion(x)
