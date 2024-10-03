import torch
import torch.nn.functional as F
from torch import nn, einsum

from typing import List, Optional, Callable, Tuple
from beartype import beartype

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from functools import partial

from classifier_free_guidance_pytorch import TextConditioner, AttentionTextConditioner, classifier_free_guidance

# from VOT.transformer import MaxViT
# from VOT.MaxViT_v2 import MaxViT_v2
from VOT.swin_transformer import SwinTransformer
from VOT.token_learner import TokenLearner
from VOT.transformer import *

import seaborn
import matplotlib.pyplot as plt

# Robotic Transformer
'''
RT-1 model involves:
1. Conditioning the input
2. Processing the images using the Vision Transformer
3. Refining the token representations
4. Applying positional embeddings 
5. Passing through a Transformer
6. Reducing the attended tokens
7. Generating the logits for action prediction.
'''

@beartype
class VTN_Swin(nn.Module):
    def __init__(
        self,
        *,
        vit: SwinTransformer,
        length = 16,
        width = 12,
        depth = 6, 
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        min_match_ratio = 0.9,
        max_distance = 0.5
    ):
        super().__init__()
        self.vit = vit

        self.token_learner = TokenLearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth
        self.min_match_ratio = min_match_ratio
        self.max_distance = max_distance

        self.transformer = Transformer(
            dim = vit.embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth
        )

        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, length * width),
            Rearrange('... (a b) -> ... a b', a = length, b = width)
        )

    # @classifier_free_guidance
    def forward(
        self,
        video
    ):
        depth = self.transformer_depth
        frames, device = video.shape[1], video.device
        video = rearrange(video, 'b f h w c-> b f c h w')
        images, packed_shape = pack_one(video, '* c h w') 
        images = images.float()

        tokens = self.vit(
            images,
            return_embeddings = True
        )
        tokens = unpack_one(tokens, packed_shape, '* c h w')
        learned_tokens = self.token_learner(tokens)
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')
        attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)
        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)
        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)
        attended_tokens = self.transformer(learned_tokens, attn_mask = ~attn_mask)
        pooled = reduce(attended_tokens, 'b (f n) d -> b d', 'mean', f = frames)
        logits = self.to_logits(pooled)
        return logits