import os
from typing import List
import numpy as np
import pandas as pd
import transformers

from .config import CONFIG

class SBICDatasetPreprocessing:
    def __init__(
            self,
            label_pad_token_id=-100,
            model_max_length=1024,
            textFields = None,
            classFields = None,
            columns_to_drop = None
        ) -> None:
            self.label_pad_token_id = label_pad_token_id
            self.model_max_length = model_max_length
            self.textFields = textFields
            self.classFields = classFields
            self.drop_train_columns = columns_to_drop


    def prepare_columns_for_train(self, df):
        df_red = df.drop(columns=self.drop_train_columns, axis=1)
        rename_cols = ['post', 'offensiveYN', 'intentYN', 'sexYN', 'whoTarget', 'targetMinority', 'targetStereotype', 'speakerMinorityYN', 'dataSource']
        df_red = df_red[rename_cols]
        df_red = df_red.rename(columns={"whoTarget": "groupTargetYN", "speakerMinorityYN": "inGroupYN", 'targetMinority': 'targetGroup'})
        return df_red
    

    def remove_nan_offensive(self, df, verbose=True):
        if verbose:
            print(df.isna().sum())

        nan_off_ann = df["offensiveYN"].isna().sum()
        if verbose:
            print(f"\nNumber of post that have NaN offensive label: {nan_off_ann} ({nan_off_ann/len(df)*100:.1f} %)")

        drop_idx = [idx for idx,val in (df["offensiveYN"].isna()).items() if val]
        df = df.drop(drop_idx)
        return df
    

    def remove_nan_group(self, df, verbose=True):
        nooff_mask = df['offensiveYN'] < 0.5
        nogrp_mask = df["groupTargetYN"].isna()

        if verbose:
            print(f"Total not offensive annotations: \t{sum(nooff_mask)}")
            print(f"Total nan values:\t\t\t{sum(nogrp_mask)}")
            print(f"Total nan values with offensive=0:\t{sum(df[nogrp_mask]['offensiveYN'] < 0.5)}")
            print(f"Total nan values with offensive=1:\t{sum(~df[nooff_mask]['groupTargetYN'].isna())}")

        drop_idx = [idx for idx,val in (~df[nooff_mask]['groupTargetYN'].isna()).items() if val]
        df = df.drop(drop_idx)
        return df


    def remove_nan_group_and_stereotype(self, df, verbose=True):
        nooff_mask = df['offensiveYN'] < 0.5
        nogrp_mask = df["groupTargetYN"] < 0.5
        tgtgrp_mask = ~df["targetGroup"].isna()

        if verbose:
            print("Analysis of targetGroup columns")
            print(f"Total not nan values if offensive=0:\t{sum(~df[nooff_mask]['targetGroup'].isna())}")
            print(f"Total not nan values if group=0:\t{sum(~df[nogrp_mask]['targetGroup'].isna())}")
            print()
            print("Analysis of targetStereotype columns")
            print(f"Total not nan values if offensive=0:\t{sum(~df[nooff_mask]['targetStereotype'].isna())}")
            print(f"Total not nan values if group=0:\t{sum(~df[nogrp_mask]['targetStereotype'].isna())}")
            print(f"Total nan values a group is set:\t{sum(df[tgtgrp_mask]['targetStereotype'].isna())}")

        drop_idx = [idx for idx,val in (~df[nooff_mask]['targetGroup'].isna()).items() if val]
        drop_idx2 = [idx for idx,val in (~df[nogrp_mask]['targetGroup'].isna()).items() if val]
        drop_idx3 = [idx for idx,val in (df[tgtgrp_mask]['targetStereotype'].isna()).items() if val]
        df = df.drop(set(drop_idx+drop_idx2+drop_idx3))

        return df
    

    def binarize_classification_features(self, df):
        for field in  self.classFields:
            df[field] = df[field].apply(lambda c: int(c>=0.5) if c < 2 else 2)
        return df
    

    def preproc_data(self, df, split):
        df_red = self.prepare_columns_for_train(df)
        df_red = self.remove_nan_offensive(df_red, verbose=False)
        df_red = self.remove_nan_group(df_red, verbose=False)
        df_red = self.remove_nan_group_and_stereotype(df_red, verbose=False)
        df_red[["groupTargetYN", "inGroupYN"]] = df_red[["groupTargetYN", "inGroupYN"]].fillna(0.5)
        df_red[["targetGroup", "targetStereotype"]] = df_red[["targetGroup", "targetStereotype"]].fillna("")
        df_red.to_pickle(os.path.join(CONFIG.dataset.preproc_dir, f"{split}.pkl"))
        print(f"Element removed during pre-processing {split} split:\t{len(df)-len(df_red)} ({(len(df)-len(df_red))/len(df):.2%})")
        return df_red
    

    def remove_overlap_data(self, df_ref, def_tgt):
        idx_to_drop = def_tgt[def_tgt["post"].isin(df_ref["post"])].index
        def_tgt = def_tgt.drop(idx_to_drop)
        return len(idx_to_drop), def_tgt


    def data_aggregator(self, df):
        aggDict = {c: lambda x: np.mean(x) for c in self.classFields}
        aggDict.update({c: lambda x: sorted(filter(lambda x: x, set(x))) for c in self.textFields})
        aggDict.update({c: lambda x: np.mean(x) for c in ["inGroupYN"]})
        aggDict.update({c: lambda x: set(x) for c in ["dataSource"]})

        agg_df = df.groupby("post", as_index=False).agg(aggDict)
        # agg_df['targetGroup'] = agg_df['targetGroup'].apply(lambda c: [m.lower().strip() for min in c for m in min.split(",")])
    
        # agg_df = self.binarize_classification_features(agg_df)
        for field in ['offensiveYN', 'intentYN', 'sexYN']:
            agg_df[field] = agg_df[field].apply(lambda c: int(c>=0.5))
        for field in ['groupTargetYN', "inGroupYN"]:
            agg_df[field] = agg_df[field].apply(lambda c: int(c>0.5) if c != 0.5 else 2)

                                    
        return agg_df