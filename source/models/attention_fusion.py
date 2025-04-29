import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for multimodal features.
    Given two inputs (audio, video), it learns weights and fuses them.
    """
    def __init__(self, audio_dim, video_dim, hidden_dim=256, num_classes=8, dropout=0.3):
        super(AttentionFusion, self).__init__()
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)
        self.video_fc = nn.Linear(video_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio_feat, video_feat):
        audio_proj = self.audio_fc(audio_feat)
        video_proj = self.video_fc(video_feat)
        concat = torch.cat([audio_proj, video_proj], dim=1)
        attn_weights = self.attention(concat)
        fused = attn_weights[:, 0].unsqueeze(1) * audio_proj + attn_weights[:, 1].unsqueeze(1) * video_proj
        return self.classifier(fused)
