import numpy as np
import torch.nn.functional as F

def generative_loss(logits, labels):
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