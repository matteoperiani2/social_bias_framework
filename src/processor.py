import numpy as np
import torch
import math
from transformers import LogitsProcessor

class RestrictClassificationTokensProcessor(LogitsProcessor):
    def __init__(self, step_cls_tokens, sep_token_id):
        self.step_cls_tokens = np.asarray(step_cls_tokens)
        self.cls_tokens = self.step_cls_tokens.flatten()
        self.sep_token_id = sep_token_id
        self.gen_token = 0

    def __call__(self, input_ids, scores):
        # Apply the restriction to only three specific tokens
        if self.gen_token < len(self.step_cls_tokens):
            restricted_tokens = self.step_cls_tokens[self.gen_token]
            mask = self._restrict_to_tokens(restricted_tokens, scores)
            self.gen_token += 1
        elif self.gen_token == len(self.step_cls_tokens):
            mask = self._restrict_to_tokens(self.sep_token_id, scores)
        else:
            mask = torch.full_like(scores, 0)
            mask[..., self.cls_tokens] = -math.inf
        return scores + mask
    
    def _restrict_to_tokens(self, valid_tokens, scores):
        mask = torch.full_like(scores, -math.inf)
        mask[..., valid_tokens] = 0
        return mask
        