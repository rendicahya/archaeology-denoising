import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=1,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        num_classes=1,
    ):
        super(ViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.num_patches = (img_size // patch_size) ** 2
        self.seq_length = self.num_patches + 1  # including class token

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embed = nn.Parameter(torch.randn(1, self.seq_length, embed_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4
            ),
            num_layers=num_layers,
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes * img_size * img_size),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # Shape: (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (B, seq_length, embed_dim)
        x += self.position_embed  # Add positional embedding

        x = self.transformer(x)  # Shape: (B, seq_length, embed_dim)
        x = x[:, 0]  # Take the class token

        x = self.mlp_head(x)  # Shape: (B, num_classes * img_size * img_size)
        x = x.view(
            B, self.in_channels, self.img_size, self.img_size
        )  # Reshape to (B, C, H, W)

        return x
