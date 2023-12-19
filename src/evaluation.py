import gc
import numpy as np
from transformers import LogitsProcessorList
from tqdm import tqdm
import torch
import pandas as pd

from .utils import get_predictions
from .processor import RestrictClassificationTokensProcessor

from sklearn.metrics import f1_score
from rouge import Rouge


def generate_predictions(model, tokenizer, dataloader, split, gen_cfg, config):  
    model.eval()
  
    clf_labels = np.zeros((len(dataloader)*config['batch_size'], 5), dtype=np.int32)
    clf_preds = np.zeros((len(dataloader)*config['batch_size'], 5), dtype=np.int32)

    minority_preds = []
    minority_labels = []
    stereotype_preds = []
    stereotype_labels = []

    allowed_tokens = ['<|offY|><|offN|>', '<|intY|><|intN|>', '<|sexY|><|sexN|>', '<|grpY|><|grpN|>', '<|ingrpY|><|ingrpN|>']
    allowed_tokens_ids = tokenizer(allowed_tokens)['input_ids']
    positive_cls_tokens = tokenizer('<|offY|><|intY|><|sexY|><|grpY|><|ingrpY|>')['input_ids']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        with tqdm(total=len(dataloader), unit="batch", desc=split) as pbar:
            for idx,data in enumerate(dataloader):
                inputs = {k: torch.as_tensor(v, device=device) for k,v in data.items() if k in ['input_ids', 'attention_mask']}

                processor = RestrictClassificationTokensProcessor(step_cls_tokens=allowed_tokens_ids,
                                                                  sep_token_id=tokenizer.sep_token_id, 
                                                                  eos_token_id=tokenizer.eos_token_id,
                                                                  max_length=gen_cfg.max_new_tokens,
                                                                  device=device)
                logits_processor = LogitsProcessorList([processor])

                generate_out = model.generate(**inputs,
                                              generation_config=gen_cfg,
                                              logits_processor=logits_processor
                )
                generate_out = generate_out.cpu().numpy()                
                gen_clf, gen_minorities, gen_stereotypes = get_predictions(tokenizer, generate_out, positive_cls_tokens)

                start_idx = idx*config['batch_size']
                end_idx = start_idx+config['batch_size']
                clf_labels[start_idx:end_idx, ...] = np.asarray(data["class_labels"])
                clf_preds[start_idx:end_idx, ...] = np.asarray(gen_clf)

                minority_preds.extend(gen_minorities)
                minority_labels.extend(data["minority_labels"])
                stereotype_preds.extend(gen_stereotypes)
                stereotype_labels.extend(data["stereotype_labels"])

                pbar.update(1)

    gc.collect()
    torch.cuda.empty_cache()

    returns = {
        "clf_preds": clf_preds.tolist(),
        'minority_preds': minority_preds,
        'stereotype_preds': stereotype_preds,
        'clf_labels': clf_labels.tolist(),
        'minority_labels': minority_labels,
        'stereotype_labels': stereotype_labels
    }

    return pd.DataFrame.from_dict(returns)


def evaluate_classification(tokenizer, labels, predictions):
    pad_token = tokenizer.pad_token_id
    bin_labels = [(max(l[l!=pad_token]), min(l[l!=pad_token])) for l in labels]

    f1_scores = []
    for lbls, preds, bin_lbl in zip(labels, predictions, bin_labels):
        mask_pad_label = lbls != pad_token
        preds = preds[mask_pad_label]
        preds = np.where(preds == bin_lbl[1], bin_lbl[1], [bin_lbl[0]])
        f1 = f1_score(lbls[mask_pad_label],
                      preds,
                      pos_label=bin_lbl[1],
                      average="binary")
        f1_scores.append(f1)

    return f1_scores


def evaluate_generation(labels, predictions):  
    rouge_metric = Rouge()
    rouge_scores = []
    for lbls, preds in zip(labels, predictions):
        # if post is offensive or it target a group
        if len(lbls) > 0 or preds != '':
            # just for can evaluate train set that is not aggregated... REMOVE BEFORE SUBMIT!!!
            if isinstance(lbls, list):
                r_scores = []
                for lbl in lbls:
                    r_score = rouge_metric.get_scores(preds, lbl)[0]["rouge-l"]["f"]
                    r_scores.append(r_score)
                rouge_scores.append(np.nan_to_num(np.max(r_scores)))
            else:
                r_score = rouge_metric.get_scores(preds, lbls)[0]["rouge-l"]["f"]
                rouge_scores.append(np.nan_to_num(r_score))

    return sum(rouge_scores)/len(rouge_scores)