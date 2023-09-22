import numpy as np
import pandas as pd
from collections import Counter
import os

data_folder = "data"
raw_folder = "raw"
aggregate_folder = "aggregated"

textFields = ['targetMinority','targetCategory', 'targetStereotype']
classFields = ['whoTarget', 'intentYN', 'sexYN','offensiveYN']

for split in ["train", "validation", "test"]:
    df = pd.read_csv(os.path.join(os.getcwd(), data_folder, raw_folder, f"{split}.csv"))
    aggDict = {c: lambda x: sorted(filter(lambda x: x, list(x))) for c in textFields}

    aggDict.update({c: lambda x: np.mean(x) for c in classFields})
    df[textFields] = df[textFields].fillna("")

    agg_df = df.groupby("post",as_index=False).agg(aggDict)
    agg_df["hasBiasedImplication"] = (agg_df["targetStereotype"].apply(len) == 0).astype(int)
    
    agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: dict(Counter(c)))
    agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: {k: round(v / total,2) for total in (sum(c.values(), 0.0),) for k, v in c.items()})

    agg_df.to_pickle(os.path.join(os.getcwd(), data_folder, aggregate_folder, f"{split}.pkl"))

    