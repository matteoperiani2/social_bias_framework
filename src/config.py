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
        batch_size:int = 4
        max_sequence_length:int = 800
        device:str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    dataset: Dataset = Dataset()
    train_params: TrainParameters = TrainParameters()
    seed:int = 42
