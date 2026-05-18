"""
BlinkerNet: a model designed specifically for short blinker-sequence videos.

Key design choices based on dataset characteristics:
  - Short videos (~56 frames) with sparse bright signal (blinker on/off)
  - Discriminative info = temporal pattern + which side of vehicle (left/right)
  - Small dataset (~14 train samples per class) → keep model small

Strategy:
  1. Per-frame 2D encoder → spatial feature vector (preserves left/right info)
  2. 1D temporal conv stack → temporal pattern recognition (no aggressive pooling)
  3. Attention over time → focuses on frames where blinker fires
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    """Lightweight per-frame 2D encoder: (1, 256, 256) → (feat_dim,)"""
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 7, stride=2, padding=3),     # 16 x 128x128
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # 16 x 64x64
            nn.Conv2d(16, 32, 5, stride=2, padding=2),    # 32 x 32x32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),    # 64 x 16x16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 128 x 8x8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                       # 128 x 4x4
            nn.Flatten(),
            nn.Linear(128 * 16, feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class TemporalAttentionHead(nn.Module):
    """Learn which frames matter (frames where blinker fires get high weight)."""
    def __init__(self, feat_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, D)
        scores = self.attn(x).squeeze(-1)                  # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   # (B, T, 1)
        pooled = (x * weights).sum(dim=1)                  # (B, D)
        return pooled, weights.squeeze(-1)


class BlinkerNet(nn.Module):
    """
    Input:  (B, 1, T, H, W)  — grayscale video, e.g. T=64, H=W=256
    Output: (B, num_classes)
    """
    def __init__(self, num_classes, feat_dim=128, use_frame_diff=True):
        super().__init__()
        self.use_frame_diff = use_frame_diff
        in_ch = 2 if use_frame_diff else 1  # original + diff stacked as channels

        # Override first conv of FrameEncoder if using diff
        self.encoder = FrameEncoder(feat_dim=feat_dim)
        if use_frame_diff:
            self.encoder.net[0] = nn.Conv2d(2, 16, 7, stride=2, padding=3)

        # Temporal stack — keep length, learn local timing patterns
        self.temporal = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, 5, padding=2),
            nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
            nn.Conv1d(feat_dim, feat_dim, 5, padding=2),
            nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
        )

        self.attn = TemporalAttentionHead(feat_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, T, H, W)
        B, C, T, H, W = x.shape

        if self.use_frame_diff:
            # Compute frame differences to emphasize transitions (blink edges)
            diff = torch.zeros_like(x)
            diff[:, :, 1:] = (x[:, :, 1:] - x[:, :, :-1]).abs()
            x_in = torch.cat([x, diff], dim=1)  # (B, 2, T, H, W)
        else:
            x_in = x

        # Reshape to per-frame: (B*T, C_in, H, W)
        x_in = x_in.permute(0, 2, 1, 3, 4).contiguous().view(B * T, -1, H, W)

        # Per-frame features
        feats = self.encoder(x_in)                          # (B*T, D)
        feats = feats.view(B, T, -1)                        # (B, T, D)

        # Temporal modeling
        feats_t = feats.permute(0, 2, 1)                    # (B, D, T)
        feats_t = self.temporal(feats_t)
        feats_t = feats_t.permute(0, 2, 1)                  # (B, T, D)

        pooled, _ = self.attn(feats_t)                      # (B, D)
        return self.classifier(pooled)
