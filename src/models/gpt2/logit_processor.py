import numpy as np
import torch
from transformers import LogitsProcessor


class GPT2RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, step_cls_tokens, sep_token_id, eos_token_id, max_length, device):
        self.step_cls_tokens = torch.as_tensor(step_cls_tokens).to(device)
        self.cls_tokens = torch.as_tensor(np.asarray(step_cls_tokens).flatten()).to(
            device
        )
        self.sep_token_id = torch.tensor(sep_token_id).to(device)
        self.eos_token_id = torch.tensor(eos_token_id).to(device)
        self.gen_token = torch.tensor(0).to(device)
        self.max_length = max_length
        self.inf = torch.finfo(torch.float16).min
        self.device = device

    def __call__(self, input_ids, scores):
        # shape input_ids: (B, SL)
        # shape scores: (B, VS)

        # Apply the restriction to only three specific tokens
        if torch.lt(self.gen_token, self.step_cls_tokens.shape[0] - 1):
            restricted_tokens = self.step_cls_tokens[self.gen_token]
            mask = self._restrict_tokens(restricted_tokens, scores)
        elif torch.eq(self.gen_token, self.step_cls_tokens.shape[0] - 1):
            mask = self._restrict_tokens(self.sep_token_id, scores)
        elif torch.eq(self.gen_token, self.max_length - 1):
            mask = self._restrict_tokens(self.eos_token_id, scores)
        else:
            eos_mask_not_offensive = (
                torch.logical_not(
                    torch.eq(input_ids[:, -self.gen_token], self.cls_tokens[0])
                )
                .type(torch.uint8)
                .reshape(scores.shape[0], 1)
            )  # (B, 1)

            seps_mask = (
                torch.logical_and(
                    torch.eq(
                        torch.sum(torch.eq(input_ids, self.sep_token_id), dim=-1), 4
                    ),
                    torch.logical_not(
                        torch.logical_or(
                            torch.eq(input_ids[:, -1], self.cls_tokens[-2]),
                            torch.eq(input_ids[:, -1], self.cls_tokens[-1]),
                        )
                    ),
                )
                .type(torch.uint8)
                .reshape(scores.shape[0], 1)
            )

            eos_mask_end = (
                torch.logical_or(
                    torch.eq(input_ids[:, -1], self.cls_tokens[-2]),
                    torch.eq(input_ids[:, -1], self.cls_tokens[-1]),
                )
                .type(torch.uint8)
                .reshape(scores.shape[0], 1)
            )

            eos_mask = self._restrict_tokens(self.eos_token_id, scores)
            eos_mask = torch.mul(
                torch.add(eos_mask_not_offensive, eos_mask_end), eos_mask
            )
            ingrp_mask = self._restrict_tokens(self.step_cls_tokens[-1], scores)
            ingrp_mask = torch.mul(ingrp_mask, seps_mask)

            mask = torch.add(eos_mask, ingrp_mask)

            mask[..., self.cls_tokens] = self.inf

        self.gen_token = torch.add(self.gen_token, 1)

        return torch.add(scores, mask)

    def _restrict_tokens(self, valid_tokens, scores):
        mask = torch.full_like(scores, self.inf)
        mask[..., valid_tokens] = 0
        return mask
