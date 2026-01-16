import torch
import torch.nn as nn


class QClassifierCNN(nn.Module):
    """
    Classifier for discrete VQ-VAE latents.

    Input:
        q : LongTensor of shape (B, H, W)
            Values in {0, ..., K-1}

    Architecture:
        q -> Embedding(K, D) -> (B, D, H, W)
          -> small CNN
          -> global average pooling
          -> Linear -> class logits
    """

    def __init__(
        self,
        n_embeds: int,      # K (size of discrete codebook)
        hidden_dim: int,    # D (embedding dimension)
        n_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Token embedding for discrete q
        self.token_embed = nn.Embedding(n_embeds, hidden_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, D, 1, 1)
            nn.Flatten(),                  # (B, D)
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: LongTensor (B, H, W)

        Returns:
            logits: FloatTensor (B, n_classes)
        """
        z = self.token_embed(q)      # (B, H, W, D)
        z = z.permute(0, 3, 1, 2)    # (B, D, H, W)
        z = self.cnn(z)              # (B, D, H, W)
        logits = self.head(z)        # (B, n_classes)
        return logits