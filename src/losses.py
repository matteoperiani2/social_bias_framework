import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


structure_tree = [
    [torch.tensor([])], #root
    [                   #level 1
        torch.tensor([50259, 50260]),
        torch.tensor([50261, 50262]),
        torch.tensor([50263, 50264]),
        torch.tensor([50265, 50266]),
        torch.tensor([50267, 50268])   
    ]
]


def default_lm_loss(logits, labels):
    labels = labels.to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def structure_loss(logits, labels):
    labels = labels.to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    probabilities = F.log_softmax(shift_logits, dim = -1)
    loss = 0

    logsumexp_all =  torch.logsumexp(shift_logits, -1, keepdim=True)
    for level_loss_list in structure_tree:

        summed_probabilities = probabilities.clone()

        for branch in level_loss_list:
            if len(branch) > 0:
                summed_probabilities[..., branch] = torch.logsumexp(shift_logits[..., branch], -1, keepdim=True) - logsumexp_all

        summed_probabilities = summed_probabilities.transpose(-1, -2)
        level_loss = F.nll_loss(summed_probabilities, shift_labels, reduction='none')

        loss += level_loss

    n_tokens_per_sample = torch.sum(shift_labels != -100, dim=-1)  # batch_size
    n_tokens_per_sample = torch.clamp(n_tokens_per_sample, min=1e-7)
    loss = torch.sum(loss, dim=-1) / n_tokens_per_sample  # batch_size
    loss = torch.mean(loss, dim=-1)

    return loss


def classification_loss(logits, labels, betas, pos_weight):
    loss = 0
    for i in range(5):
        loss = betas[i] * F.binary_cross_entropy_with_logits(input=logits[i],
                                                             target=labels[i],
                                                             pos_weight=pos_weight[i])
    return loss


class StructureLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)