from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
   

def gpt2_llm_loss(logits, labels):
    logits = logits.transpose(-1, -2)
    loss = F.cross_entropy(logits, labels, reduction='none') #(BATCH, SEQ_LEN)

    n_valid_tokens = torch.sum(labels != -100, dim=-1) # (BATCH, ) 
    loss = torch.sum(loss, dim=-1) / torch.clamp(n_valid_tokens, min=1e-7) # (BATCH, )
    loss = torch.mean(loss) # (1,)
    return loss


def classification_loss(logits, labels):
    weight = (labels!=-100).float()
    loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float(), weight=weight)
    return loss


# class ClassificationLoss(nn.Module):

#     def __init__(self,
#                  weights=torch.ones(5)
#     ):
#         super(ClassificationLoss, self).__init__()
#         self.weights = weights

#     def forward(self, logits, labels):        
#         logits = logits.transpose(-1, -2)
#         loss = F.cross_entropy(logits, labels, reduction='none') #(BATCH, SEQ_LEN)

#         n_valid_tokens = torch.sum(labels != -100, dim=-1) # (BATCH, ) 
#         loss = torch.sum(loss, dim=-1) / torch.clamp(n_valid_tokens, min=1e-7) # (BATCH, )
#         loss = torch.mean(loss) # (1,)

#         return loss

# class GenerativeLoss(nn.Module):

#     def __init__(self):
#         super(GenerativeLoss, self).__init__()
        

#     def forward(self, logits, labels):
#         logits = logits.transpose(-1, -2)
#         loss = F.cross_entropy(logits, labels, reduction='none') #(BATCH, SEQ_LEN)

#         n_valid_tokens = torch.sum(labels != -100, dim=-1) # (BATCH, )
#         loss = torch.sum(loss, dim=-1) / torch.clamp(n_valid_tokens, min=1e-7) # (BATCH, )
#         if n_valid_tokens.sum() == 0:
#             return torch.tensor([0], device=logits.device)
#         loss = torch.mean(loss[loss!=0]) # (1,)
        
#         return loss
    

# class StructureLoss(nn.Module):

#     def __init__(self, sep_token):
#         super(StructureLoss, self).__init__()
#         assert sep_token is not None, print("SEP token cannot be None!")
#         self.sep_token = sep_token

#     def forward(self, logits, labels):
#         logits = logits.transpose(-1, -2)
#         loss = F.cross_entropy(logits, labels, reduction='none') #(BATCH, SEQ_LEN)

#         n_valid_tokens = torch.sum(labels != -100, dim=-1) # (BATCH, ) 
#         loss = torch.sum(loss, dim=-1) / torch.clamp(n_valid_tokens, min=1e-7) # (BATCH, )
#         loss = torch.mean(loss) # (1,) 

#         return loss


# class CustomLoss(nn.Module):

#     def __init__(self,
#                  betas:List,
#                  sep_token = None,
#                  weights = None
#     ):
#         super(CustomLoss, self).__init__()
#         self.c_weight, self.g_weight, self.s_weight = betas
#         self.classification_loss = ClassificationLoss(weights=weights)
#         self.generative_loss = GenerativeLoss()
#         self.structure_loss = StructureLoss(sep_token=sep_token)

    
#     def forward(self, logits, classification_labels, generative_labels):
#         c_loss = self.classification_loss(logits, classification_labels)
#         g_loss = self.generative_loss(logits, generative_labels)
#         # s_loss = self.structure_loss(logits, structure_labels)

#         loss = self.c_weight * c_loss + self.g_weight * g_loss 

#         return loss, (c_loss.item(), g_loss.item())

#         return loss
# class CEwithStructureImportance(nn.Module):
#     def __init__(self, alpha, weight=None):
#         super(CEwithStructureImportance, self).__init__()
#         self.alpha = alpha
#         self.weights = weight
#         self.class_tokens_group = [                   #level 1
#             torch.tensor([50259, 50260]),
#             torch.tensor([50261, 50262]),
#             torch.tensor([50263, 50264]),
#             torch.tensor([50265, 50266]),
#             torch.tensor([50267, 50268])   
#         ]

#     def forward(self, logits, labels):
#         labels = labels.to(logits.device)
#         shift_logits = logits[..., :-1, :]
#         shift_labels = labels[..., 1:]

#         # Classic Cross Entropy on output logits
#         ce_loss = F.cross_entropy(torch.reshape(shift_logits, (shift_logits.size(0)*shift_logits.size(1), -1)),
#                                   torch.reshape(shift_labels, (-1,)))

#         # Cross Entropy with group balanced class probabilities
#         probabilities = F.log_softmax(shift_logits, dim = -1).clone()
#         logsumexp_all =  torch.logsumexp(shift_logits, -1, keepdim=True)
#         for class_tokens in self.class_tokens_group:
#             probabilities[..., class_tokens] = torch.logsumexp(shift_logits[..., class_tokens], -1, keepdim=True) - logsumexp_all

#         probabilities = probabilities.transpose(-1, -2)
#         struct_loss = F.nll_loss(probabilities, shift_labels, reduction='none')

#         n_tokens_per_sample = torch.sum(shift_labels != -100, dim=-1) 
#         struct_loss = torch.sum(struct_loss, dim=-1) / torch.clamp(n_tokens_per_sample, min=1e-7)
#         struct_loss = torch.mean(struct_loss, dim=-1)

#         loss = ce_loss + self.alpha * struct_loss

#         return loss 