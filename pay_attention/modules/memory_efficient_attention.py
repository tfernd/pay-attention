from __future__ import annotations
from typing import Optional

import math

import torch
from torch import Tensor


def memory_efficient_attention(
    query: Tensor,  # (B, T, C)
    key: Tensor,  # (B, T', C)
    value: Tensor,  # (B, T', C')
    mask: Optional[Tensor] = None,  # shape(B,? T, T')
    /,
    *,
    query_seq_chunk_size: int = 1_024,
    key_seq_chunk_size: int = 4_096,
) -> Tensor:  # (B, T, C')
    B, T, C = query.shape
    B, Tp, Cp = value.shape

    # scale query and key
    scale = math.pow(C, -1 / 4)
    query = query * scale
    key = key * scale

    out = query.new_empty(B, T, Cp)  # (B, T, C')
    for i in range(0, T, query_seq_chunk_size):
        q_slice = slice(i, min(i + query_seq_chunk_size, T))

        unorm_outs: list[Tensor] | Tensor = []
        unorm_attn_sums: list[Tensor] | Tensor = []
        max_scores: list[Tensor] | Tensor = []
        for j in range(0, Tp, key_seq_chunk_size):
            key_slice = slice(j, min(j + key_seq_chunk_size, Tp))

            key_chunk = key[:, key_slice]  # (B, key_seq_chunk_size, C)
            value_chunk = value[:, key_slice]  # (B, key_seq_chunk_size, C')

            # unormalized attention computation
            # (B, query_seq_chunk_size, key_seq_chunk_size)
            score = query[:, q_slice] @ key_chunk.transpose(-1, -2)
            del key_chunk

            if mask is not None:
                mask_chunk = mask[..., q_slice, key_slice]

                if mask.dtype == torch.bool:
                    score.masked_fill_(~mask_chunk, float("-inf"))
                else:
                    score += mask_chunk

            max_score = score.amax(dim=-1)  # (B, query_seq_chunk_size)
            # (B, query_seq_chunk_size, key_seq_chunk_size)
            unorm_attn = torch.exp(score - max_score[..., None])
            del score

            unorm_attn_sum = unorm_attn.sum(dim=-1)  # (B, query_seq_chunk_size)
            unorm_out = unorm_attn @ value_chunk  # (B, query_seq_chunk_size, C')
            del unorm_attn, value_chunk

            # save chunked results
            unorm_outs.append(unorm_out)
            unorm_attn_sums.append(unorm_attn_sum)
            max_scores.append(max_score)

        # recombine chunks
        unorm_outs = torch.stack(unorm_outs, dim=1)  # (B, Tp//key_seq_chunk_size, query_seq_chunk_size, C')
        # (B, Tp//key_seq_chunk_size, query_seq_chunk_size)
        unorm_attn_sums = torch.stack(unorm_attn_sums, dim=1)
        max_scores = torch.stack(max_scores, dim=1)  # (B, Tp//key_seq_chunk_size, query_seq_chunk_size)

        global_max_score = max_scores.amax(dim=1)  # (B, query_seq_chunk_size)
        # (B, Tp//key_seq_chunk_size, query_seq_chunk_size)
        max_diffs = torch.exp(max_scores - global_max_score[:, None])
        del max_scores, global_max_score

        unorm_outs *= max_diffs[..., None]  # (B, Tp//key_seq_chunk_size, query_seq_chunk_size, C')
        unorm_attn_sums *= max_diffs  # (B, Tp//key_seq_chunk_size, query_seq_chunk_size)
        del max_diffs

        all_values = unorm_outs.sum(dim=1)  # (B, query_seq_chunk_size, C')
        del unorm_outs

        all_unorm_attn_sums = unorm_attn_sums.sum(dim=1)  # (B, query_seq_chunk_size)

        out[:, q_slice] = all_values / all_unorm_attn_sums[..., None]  # (B, query_seq_chunk_size, C')
        del unorm_attn_sums, all_values, all_unorm_attn_sums

    return out
