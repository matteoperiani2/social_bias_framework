from typing import Optional

import torch
import torch.nn.functional as F


def _binary_to_categorical(values, use_logsigmoid=False, dim=-1):
    if use_logsigmoid:
        p = F.logsigmoid(values)
        not_p = F.logsigmoid(-values)
    else:
        p = values
        not_p = 1 - values
    return torch.stack((not_p, p), dim=dim)


def binary_kl_div_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    r"""The `Kullback-Leibler divergence Loss
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Predicted unnormalized logits;
            see Shape section below for supported shapes.
        target: Tensor of the same shape as input. See :attr:`log_target` for
            the target's interpretation.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of shape (2, ), (2, d1, ) ... , (2, d1, ...)
            or broadcastable to one of these shapes
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.
            Default: -100
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be averaged
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            Default: ``'mean'``
    Shape:
        - Input: Shape :math:`(N, )` or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
    """
    input = _binary_to_categorical(input, use_logsigmoid=True, dim=1)

    mask = target != ignore_index
    mask_ = mask.unsqueeze(dim=1).repeat_interleave(2, dim=1)
    target = _binary_to_categorical(target, dim=1)
    target[~mask_] = 0.0

    loss = F.kl_div(
        input=input,
        target=target,
        reduction="none",
        log_target=False,
    )  # batch_size x n_classes x d1 x ...

    if weight is None:
        weight = torch.tensor([1.0])
    weight = weight.to(input.device)
    loss = torch.einsum("bj...,j...->b...", loss, weight)  # batch_size x d1 x ...

    if reduction == "none":
        return loss

    loss = loss.sum()  # (1, )
    if reduction == "sum":
        return loss

    if reduction == "batchmean":
        mask = torch.einsum("b...->b", mask).bool()  # batch_size

    return loss / mask.sum()
