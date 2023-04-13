import torch
from torch.autograd import Function

import standard_attention_cuda as sac


class StandardAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, mask):
        B, T, C = q.size()
        B, T_, C_ = v.size()

        out = torch.empty((B, T, C_), dtype=q.dtype, device=q.device)

        sac.forward(q, k, v, mask, out)

        return out


def standard_attention_cuda(q, k, v, mask=None):
    if mask is None:
        mask = torch.tensor([], dtype=torch.bool, device=q.device)
    else:
        mask = mask.to(dtype=torch.bool, device=q.device)

    return StandardAttentionFunction.apply(q, k, v, mask)
