from __future__ import annotations

from tqdm.auto import trange

import torch
from torch import Tensor

from ..utils import scaled


def memory_efficient(
    q: Tensor,  # (B, T, C)
    k: Tensor,  # (B, T', C)
    v: Tensor,  # (B, T', C')
    *,
    q_chunks: int = 1_024,
    k_chunks: int = 4_096,
) -> Tensor:  # (B, T, C')
    """
    This function computes the scaled dot product attention between a set of queries, keys and values using a memory-efficient algorithm. The implementation is almost identical to the one proposed in the paper, with additional masking and adding bias compatibility, batch dimensions support and PyTorch implementation.
    For computing attention, the proposed method requires only O(sqrt(n)) memory, making it a drop-in replacement for attention calculation that provides significant memory savings.
    To optimize for memory consumption and runtime, the function provides the `key_chunk_size` and `query_chunk_size` parameters which should be adjusted to the best configuration for the specific use case.
    """
    
    B, T, C = q.shape
    B, Tp, Cp = v.shape

    out = torch.empty(B, T, Cp, device=q.device, dtype=q.dtype)
    for i in trange(0, T, q_chunks):
        si = slice(i, min(i + q_chunks, T))

        qc = scaled(q[:, si])  # (B, q_chunks, C)

        unorm_outs: list[Tensor] | Tensor = []
        unorm_attn_sums: list[Tensor] | Tensor = []
        max_scores: list[Tensor] | Tensor = []
        for j in range(0, Tp, k_chunks):
            sj = slice(j, min(j + k_chunks, Tp))

            kc = k[:, sj]  # (B, k_chunks, C)
            vc = v[:, sj]  # (B, k_chunks, C')

            # unormalized attention computation
            score = qc @ kc.transpose(-1, -2)  # (B, q_chunks, k_chunks)
            del kc
            max_score = score.amax(dim=-1)  # (B, q_chunks)
            unorm_attn = torch.exp(score - max_score[..., None])  # (B, q_chunks, k_chunks)
            del score
            unorm_attn_sum = unorm_attn.sum(dim=-1)  # (B, q_chunks)
            unorm_out = unorm_attn @ vc  # (B, q_chunks, C')
            del unorm_attn, vc

            # save chunked results
            unorm_outs.append(unorm_out)
            unorm_attn_sums.append(unorm_attn_sum)
            max_scores.append(max_score)

        # recombine chunks
        unorm_outs = torch.stack(unorm_outs, dim=1)  # (B, Tp//k_chunks, q_chunks, C')
        unorm_attn_sums = torch.stack(unorm_attn_sums, dim=1)  # (B, Tp//k_chunks, q_chunks)
        max_scores = torch.stack(max_scores, dim=1)  # (B, Tp//k_chunks, q_chunks)

        global_max_score = max_scores.amax(dim=1)  # (B, q_chunks)
        max_diffs = torch.exp(max_scores - global_max_score[:, None])  # (B, Tp//k_chunks, q_chunks)
        del max_scores, global_max_score

        unorm_outs *= max_diffs[..., None]  # (B, Tp//k_chunks, q_chunks, C')
        unorm_attn_sums *= max_diffs  # (B, Tp//k_chunks, q_chunks)
        del max_diffs

        all_values = unorm_outs.sum(dim=1)  # (B, q_chunks, C')
        del unorm_outs
        all_unorm_attn_sums = unorm_attn_sums.sum(dim=1)  # (B, q_chunks)

        out[:, si] = all_values / all_unorm_attn_sums[..., None]  # (B, q_chunks, C')
        del unorm_attn_sums, all_values, all_unorm_attn_sums

    return out
