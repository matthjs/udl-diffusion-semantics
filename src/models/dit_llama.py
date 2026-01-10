"""
Base implementation taken from: https://github.com/cloneofsimo/d3pm
Vision Transformer for images (CIFAR-10)
"""


# Code heavilty based on https://github.com/Alpha-VLLM/LLaMA2-Accessory

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ditutil import FinalLayer, LabelEmbedder, TimestepEmbedder, TransformerBlock

class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        N=8,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
        learn_sigma=True,
    ):
        super().__init__()
        self.N = N
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = N * in_channels * 2
        self.input_size = input_size
        self.patch_size = patch_size

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x_onehot = torch.nn.functional.one_hot(x, self.N).float().to(x.device)
        x = (2 * x.float() / (self.N - 1)) - 1.0
        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = t + y

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        x, gate = (
            x.reshape(x.shape[0], -1, self.N * 2, *x.shape[2:])
            .transpose(2, -1)
            .contiguous()
        ).chunk(2, dim=-1)

        return x + x_onehot * (1 + gate).abs()

        # x = (x.reshape(x.shape[0], -1, self.N, *x.shape[2:])
        #     .transpose(2, -1)
        #     .contiguous()
        # )
        # return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=256, n_layers=16, n_heads=32, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)
