"""
Base implementation taken from: https://github.com/cloneofsimo/d3pm
1D Transformer for discrete sequences (text)
"""


# Code heavilty based on https://github.com/Alpha-VLLM/LLaMA2-Accessory

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit_llama import DiT_Llama
from models.ditutil import FinalLayer, TimestepEmbedder, TransformerBlock

class DDiT_Llama(nn.Module):

    def __init__(
        self,
        N=256,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        learn_gating=False,
    ):
        super().__init__()
        self.N = N
        self.learn_gating = learn_gating
        if self.learn_gating:
            self.out_channel = N * 2
        else:
            self.out_channel = N

        self.embedder = nn.Embedding(N, dim)
        self.t_embedder = TimestepEmbedder(min(dim, 1024))
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
        self.final_layer = FinalLayer(dim, 1, self.out_channel)
        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

    def forward(self, x, t, cond=None):
        self.freqs_cis = self.freqs_cis.to(x.device)
        x_onehot = torch.nn.functional.one_hot(x, self.N).to(
            x.device, dtype=next(self.parameters()).dtype
        )
        x = self.embedder(x)
        adaln_input = self.t_embedder(t)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        if self.learn_gating:
            x, gate = x.chunk(2, dim=-1)
            return x + x_onehot * (1 + gate).abs()
        else:
            return x + x_onehot

