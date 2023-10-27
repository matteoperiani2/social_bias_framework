import numpy as np
from rouge import Rouge
from sklearn.metrics import f1_score

from .config import CONFIG

rouge_metric = Rouge()

def evaluate_predictions(tokenizer, predictions, labels):
    class_labels = reshape_tokens_for_metric(labels["class_labels"])
    class_preds, minority_preds, stereotype_preds = get_predictions(tokenizer, predictions)

    f1_classifiaction = eval_classification_tokens(tokenizer, class_labels, class_preds)
    f1_rouge_minority = eval_generation_tokens(labels["minority_labels"], minority_preds)
    f1_rouge_stereotype = eval_generation_tokens(labels["stereotype_labels"], stereotype_preds)

    return {
        "class_f1": f1_classifiaction,
        "rouge_minor": f1_rouge_minority,
        "rouge_strtp": f1_rouge_stereotype
        # "rouge_minor": [0],
        # "rouge_strtp": [0]
    }


def reshape_tokens_for_metric(tokens):
    n_class = 5
    n_lbls = len(tokens)

    class_labels = []
    for i in range(n_class):
        for j in range(n_lbls):
            try:
                class_labels.append(tokens[j][i])
            except:
                class_labels.append(-1)
    
    return np.reshape(class_labels, (n_class, n_lbls))


def get_predictions(tokenizer, predictions):
    class_preds = []
    minority_preds = []
    stereotype_preds = []

    for pred in predictions:
        sep_idx = np.where(pred == tokenizer.sep_token_id)[0]
        eos_idx = np.where(pred == tokenizer.eos_token_id)[0]
        
        # --- get classification tokens --- 
        if len(eos_idx) > 0:
            # concatenate first 4 tokens with the token generated before the eos
            class_preds.append(np.concatenate((pred[:4], [pred[eos_idx[0]-1]])))
        else:
            # concatenate first 4 tokens with the last generated token
            class_preds.append(np.concatenate((pred[:4], [pred[-1]])))
        
        # --- get minority and stereotype tokens ---
        if len(sep_idx) > 0:
            ## MINORITY
            if len(sep_idx) > 1: 
                # select as minority tokens, those tokens that are between first 2 sep
                minority_preds.append(pred[sep_idx[0]+1:sep_idx[1]])

                ## STEREOTYPE
                if len(sep_idx) > 2:
                    # select as stereotype tokens, those tokens that are between the 2nd and the 3rd sep
                    stereotype_preds.append(pred[sep_idx[1]+1:sep_idx[2]])
                else:
                    # select as stereotype tokens, those tokens that are after the 2nd sep
                    stereotype_preds.append(pred[sep_idx[1]+1:])
            else:
                # select as minority tokens, those tokens that are after the sep
                # for stereotypes no tokens are selected
                minority_preds.append(pred[sep_idx[0]+1:])
                stereotype_preds.append([])
        else:
            # in the case the output is very bad, both minority and stereotypes are discarded
            minority_preds.append([])
            stereotype_preds.append([])


    class_preds = reshape_tokens_for_metric(class_preds)
    minority_preds = tokenizer.batch_decode(minority_preds)
    stereotype_preds = tokenizer.batch_decode(stereotype_preds)

    return  class_preds, minority_preds, stereotype_preds


def eval_classification_tokens(tokenizer, labels, predictions):
    f1_scores = []
    for lbls, preds in zip(labels, predictions):
        good_idx = [idx for idx,lbl in enumerate(lbls) if lbl != tokenizer.pad_token_id]
        f1 = f1_score([lbls[index] for index in good_idx],
                      [preds[index] for index in good_idx],
                       zero_division=0,
                       average="macro")
        f1_scores.append(np.nan_to_num(f1))
    return f1_scores


def eval_generation_tokens(labels, predictions):
    rouge_score = []
    for lbls, preds in zip(labels, predictions):
        if len(lbls) > 0:
            if preds != '':
                lbl_score = []
                for lbl in lbls:
                    r_score = rouge_metric.get_scores(preds, lbl)[0]["rouge-l"]["f"]
                    lbl_score.append(r_score)

                rouge_score.append(np.nan_to_num(np.mean(lbl_score)))
            else:
                rouge_score.append(0.)
            
    return rouge_score