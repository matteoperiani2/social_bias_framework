from typing import List
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import os
import torch

from .config import Config

CONFIG: Config = Config()

def data_aggregator(splits:List):
    textFields = ['targetMinority','targetCategory', 'targetStereotype']
    classFields = ['whoTarget', 'intentYN', 'sexYN','offensiveYN']

    for split in splits:
        df = pd.read_pickle(os.path.join(CONFIG.dataset.preproc_dir, f"{split}.pkl"))
        aggDict = {c: lambda x: sorted(filter(lambda x: x, list(x))) for c in textFields}

        aggDict.update({c: lambda x: np.mean(x) for c in classFields})
        df[textFields] = df[textFields].fillna("")

        agg_df = df.groupby("post",as_index=False).agg(aggDict)
        agg_df["hasBiasedImplication"] = (agg_df["targetStereotype"].apply(len) == 0).astype(int)
        
        agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: [m.lower().strip() for min in c for m in min.split(",")])
        # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: dict(Counter(c)))
        # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: {k: round(v / total,2) for total in (sum(c.values(), 0.0),) for k, v in c.items()})

        agg_df.to_pickle(os.path.join(CONFIG.dataset.aggregated_dir, f"{split}.pkl"))

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def aggregate_minority(min_counter:Counter, min_emb:dict, tresh=0.9):
    sim_minorities = defaultdict(list)
    minorities_map = {}
    min_counter2 = Counter()
    
    while any(min_counter):
        min1,_ = min_counter.most_common(1)[0]
        emb1 = min_emb[min1]
        for min2,emb2 in min_emb.copy().items():
            if cosine_similarity(emb1, emb2) >= tresh:
                if min1 != min2:
                    min_counter.update({min1: min_counter.pop(min2)})
                    min_emb.pop(min2)
                sim_minorities[min1].append(min2)
                minorities_map[min2] = min1
                    
        min_counter2.update({min1: min_counter.pop(min1)})

    return min_counter2, min_emb, sim_minorities, minorities_map