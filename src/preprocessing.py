import os
from typing import List
import numpy as np
import pandas as pd
import transformers

from .config import CONFIG

class SBICDatasetPreprocessing:
    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer = None,
            label_pad_token_id=-100,
            model_max_length=1024,
            textFields = None,
            classFields = None
        ) -> None:
            self.tokenizer = tokenizer
            self.label_pad_token_id = label_pad_token_id
            self.model_max_length = model_max_length
            self.textFields = textFields
            self.classFields = classFields

    def create_data_for_training(self, df, drop_columns:List):
        df = df.drop(columns=drop_columns, axis=1)
        df[self.textFields] = df[self.textFields].fillna("")
        for field in  self.classFields:
            df[field] = df[field].apply(lambda c: float(c>=0.5))
        df["speakerMinorityYN"] = df["speakerMinorityYN"].fillna(0)
        
        return df

    def data_aggregator(self, splits:List):
        for split in splits:
            df = pd.read_pickle(os.path.join(CONFIG.dataset.preproc_dir, f"{split}.pkl"))
            aggDict = {c: lambda x: sorted(filter(lambda x: x, list(x))) for c in self.textFields}

            aggDict.update({c: lambda x: np.mean(x) for c in self.classFields})
            df[self.textFields] = df[self.textFields].fillna("")

            agg_df = df.groupby("post",as_index=False).agg(aggDict)
            agg_df["hasBiasedImplication"] = (agg_df["targetStereotype"].apply(len) == 0).astype(int) # ???
            
            agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: [m.lower().strip() for min in c for m in min.split(",")])
            # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: dict(Counter(c)))
            # agg_df['targetMinority'] = agg_df['targetMinority'].apply(lambda c: {k: round(v / total,2) for total in (sum(c.values(), 0.0),) for k, v in c.items()})

            agg_df.to_pickle(os.path.join(CONFIG.dataset.aggregated_dir, f"{split}.pkl"))