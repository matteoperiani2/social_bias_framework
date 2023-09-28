import os
import torch

class Config:
    
    class Dataset():
        
        data_dir: str = "data"
        
        raw_dir: str = os.path.join(data_dir, "raw")
        train_data_raw: str = os.path.join(raw_dir, "train.pkl")
        val_data_raw: str = os.path.join(raw_dir, "validation.pkl")
        test_data_raw: str = os.path.join(raw_dir, "test.pkl")

        preproc_dir: str = os.path.join(data_dir, "preproc")
        train_data_preproc: str = os.path.join(preproc_dir, "train.pkl")
        val_data_preproc:str = os.path.join(preproc_dir, "validation.pkl")
        test_data_preproc: str = os.path.join(preproc_dir, "test.pkl")

        aggregated_dir: str = os.path.join(data_dir, "aggregated")
        train_data_agg: str = os.path.join(aggregated_dir, "train.pkl")
        val_data_agg: str = os.path.join(aggregated_dir, "validation.pkl")
        test_data_agg: str = os.path.join(aggregated_dir, "test.pkl")

    class TrainParameters():
        models:str = "gpt2"
        epochs:int = 3
        batch_size:int = 2
        lr:int = 1e-5
        max_sequence_length:int = 800
        device:str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        special_tokens = {
            'bos_token': "[STR]",
            "eos_token": "[END]",
            'pad_token': '[PAD]',
            'sep_token': '[SEP]',
            "additional_special_tokens": ["[offY]", "[offN]", "[sexY]", "[sexN]", "[intY]", 
                                        "[intN]", "[grpY]", "[grpN]", "[ingrpN]", "[ingrpY]"]
        }  

    class Utils():
        label_encoder = {
            0: {0: "[grpN]", 1: "[grpY]"},
            1: {0: "[intN]", 1: "[intY]"},
            2: {0: "[sexN]", 1: "[sexY]"},
            3: {0: "[offN]", 1: "[offY]"},
            4: {0: "[ingrpN]", 1: "[ingrpY]"},
        }
        
    dataset: Dataset = Dataset()
    train_params: TrainParameters = TrainParameters()
    utils: Utils = Utils()
    seed:int = 42
