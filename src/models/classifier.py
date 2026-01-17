import torch
import torch.nn as nn


def _gn(num_channels: int, num_groups: int = 32):
    num_groups = min(num_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class _ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _gn(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _gn(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class QClassifierCNN(nn.Module):
    """
    ResNet-style classifier for discrete VQ-VAE latents (q).

    Input: q (B, H, W) int64
      q -> Embedding(K, D) -> (B, D, H, W)
        -> 1x1 projection to width
        -> residual conv blocks
        -> global avg pool -> linear logits
    """
    def __init__(
        self,
        n_embeds: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float = 0.1,
        width: int = 256,
        depth: int = 6,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(n_embeds, hidden_dim)

        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim, width, kernel_size=1, bias=False),
            _gn(width),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(*[_ResBlock(width, dropout=dropout) for _ in range(depth)])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width, n_classes),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        z = self.token_embed(q)                 # (B, H, W, D)
        z = z.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        z = self.proj(z)                        # (B, width, H, W)
        z = self.blocks(z)
        return self.head(z)