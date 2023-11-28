import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CEwithStructureImportance(nn.Module):
    def __init__(self, alpha, weight=None) -> None:
        super(CEwithStructureImportance, self).__init__()
        self.alpha = alpha
        self.weights = weight
        self.class_tokens_group = [                   #level 1
            torch.tensor([50259, 50260]),
            torch.tensor([50261, 50262]),
            torch.tensor([50263, 50264]),
            torch.tensor([50265, 50266]),
            torch.tensor([50267, 50268])   
        ]

    def forward(self, logits, labels):
        labels = labels.to(logits.device)
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        # Classic Cross Entropy on output logits
        ce_loss = F.cross_entropy(torch.reshape(shift_logits, (shift_logits.size(0)*shift_logits.size(1), -1)),
                                  torch.reshape(shift_labels, (-1,)))

        # Cross Entropy with group balanced class probabilities
        probabilities = F.log_softmax(shift_logits, dim = -1).clone()
        logsumexp_all =  torch.logsumexp(shift_logits, -1, keepdim=True)
        for class_tokens in self.class_tokens_group:
            probabilities[..., class_tokens] = torch.logsumexp(shift_logits[..., class_tokens], -1, keepdim=True) - logsumexp_all

        probabilities = probabilities.transpose(-1, -2)
        struct_loss = F.nll_loss(probabilities, shift_labels, reduction='none')

        n_tokens_per_sample = torch.sum(shift_labels != -100, dim=-1) 
        struct_loss = torch.sum(struct_loss, dim=-1) / torch.clamp(n_tokens_per_sample, min=1e-7)
        struct_loss = torch.mean(struct_loss, dim=-1)

        loss = ce_loss + self.alpha * struct_loss

        return loss 


def default_lm_loss(logits, labels):
    labels = labels.to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def classification_loss(logits, labels, betas, pos_weight):
    loss = 0
    for i in range(5):
        loss = betas[i] * F.binary_cross_entropy_with_logits(input=logits[i],
                                                             target=labels[i],
                                                             pos_weight=pos_weight[i])
    return loss