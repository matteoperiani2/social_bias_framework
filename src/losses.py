import torch
import torch.nn.functional as F


def kl_div(
    input: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "batchmean",
    log_target: bool = False,
) -> torch.Tensor:
    r"""The `Kullback-Leibler divergence Loss
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Tensor of arbitrary shape in log-probabilities.
        target: Tensor of the same shape as input. See :attr:`log_target` for
            the target's interpretation.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.
            Default: -100
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'mean'``
        log_target (bool): A flag indicating whether ``target`` is passed in the log space.
            It is recommended to pass certain distributions (like ``softmax``)
            in the log space to avoid numerical issues caused by explicit ``log``.
            Default: ``False``

    .. warning::
        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.
    """

    mask = target != ignore_index
    masked_target = target.clone()
    masked_target[~mask] = torch.finfo(input.dtype).min if log_target else 0.0

    loss = F.kl_div(
        input=input, target=masked_target, reduction="none", log_target=log_target
    )

    if reduction == "none":
        return loss

    loss = torch.sum(loss)
    if reduction == "sum":
        return loss

    if reduction == "batchmean":
        mask = torch.any(mask, dim=-1)

    return loss / mask.sum()
