from typing import List
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import os
import torch
from transformers import AutoTokenizer, AutoModel

from .config import Config

CONFIG: Config = Config()

def raw_data_cleaner(splits):
    for split in splits:
        df = pd.read_csv(f"data/old_raw/{split}.csv")
        textFields = ['targetMinority','targetCategory', 'targetStereotype']
        classFields = ['whoTarget', 'intentYN', 'sexYN','offensiveYN', "speakerMinorityYN"]
        df = df.drop(columns=["sexReason", "annotatorGender", "annotatorMinority", "sexPhrase", "WorkerId", "HITId", "annotatorPolitics", "annotatorRace", "annotatorAge"])
        df[textFields] = df[textFields].fillna("")
        for field in classFields:
            df[field] = df[field].apply(lambda c: float(c>=0.5))
        df["speakerMinorityYN"] = df["speakerMinorityYN"].fillna(0)
        df.to_pickle(os.path.join(CONFIG.dataset.raw_dir, f"{split}.pkl"))

def data_aggregator(splits:List):
    textFields = ['targetMinority','targetCategory', 'targetStereotype']
    classFields = ['whoTarget', 'intentYN', 'sexYN','offensiveYN']

    for split in splits:
        df = pd.read_pickle(os.path.join(CONFIG.dataset.preproc_dir, f"{split}.pkl"))
        aggDict = {c: lambda x: sorted(filter(lambda x: x, list(x))) for c in textFields}

        aggDict.update({c: lambda x: np.mean(x) for c in classFields})
        df[textFields] = df[textFields].fillna("")

        agg_df = df.groupby("post",as_index=False).agg(aggDict)
        agg_df["hasBiasedImplication"] = (agg_df["targetStereotype"].apply(len) == 0).astype(int) # ???
        
        agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: [m.lower().strip() for min in c for m in min.split(",")])
        # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: dict(Counter(c)))
        # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: {k: round(v / total,2) for total in (sum(c.values(), 0.0),) for k, v in c.items()})

        agg_df.to_pickle(os.path.join(CONFIG.dataset.aggregated_dir, f"{split}.pkl"))
        
def df_aggregator(df:pd.DataFrame):
    textFields = ['targetMinority','targetCategory', 'targetStereotype']
    classFields = ['whoTarget', 'intentYN', 'sexYN','offensiveYN']
    
    aggDict = {c: lambda x: sorted(filter(lambda x: x, list(x))) for c in textFields}

    aggDict.update({c: lambda x: np.mean(x) for c in classFields})
    df[textFields] = df[textFields].fillna("")

    agg_df = df.groupby("post",as_index=False).agg(aggDict)
    agg_df["hasBiasedImplication"] = (agg_df["targetStereotype"].apply(len) == 0).astype(int) # ???
    
    agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: [m.lower().strip() for min in c for m in min.split(",")])
    # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: dict(Counter(c)))
    # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: {k: round(v / total,2) for total in (sum(c.values(), 0.0),) for k, v in c.items()})

    return agg_df

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

def remove_post_from_annotation_count(df:pd.DataFrame, ann_count:int):
    post_counts = df["post"].value_counts().reset_index()
    post_to_remove = post_counts[post_counts["count"].apply(lambda c: c == ann_count)]["post"].values.tolist()
    pruned_df = df[~df['post'].isin(post_to_remove)].reset_index()
    print("Removed posts:", len(df)-len(pruned_df))
    return pruned_df

def get_minorities_embedding(df:pd.DataFrame):
    mask = df["targetMinority"].apply(lambda x: len(x)>0) # consider only those they are offensive (have some minority target)
    off_df = df[mask]
    minorities = off_df["targetMinority"].explode().unique().tolist()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    min_counter = Counter()
    for idx, row in off_df["targetMinority"].items():
        min_counter.update(row)
    # min_counter = dict(sorted(min_counter.items(), key=lambda x:x[1], reverse=True))

    encoded_input = tokenizer(minorities, padding=True, truncation=True, max_length=100, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = embeddings.cpu().numpy()
    min_emb = dict(zip(minorities, embeddings))
    
    return min_counter, min_emb