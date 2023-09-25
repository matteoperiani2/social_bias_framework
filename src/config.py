import os

class Config:
    
    class Dataset():
        
        data_dir: str = "data"
        
        raw_dir: str = os.path.join(data_dir, "raw")
        train_data_raw: str = os.path.join(raw_dir, "train.csv")
        val_data_raw: str = os.path.join(raw_dir, "validation.csv")
        test_data_raw: str = os.path.join(raw_dir, "test.csv")

        preproc_dir: str = os.path.join(data_dir, "preproc")
        train_data_preproc: str = os.path.join(preproc_dir, "train.pkl")
        val_data_preproc:str = os.path.join(preproc_dir, "validation.pkl")
        test_data_preproc: str = os.path.join(preproc_dir, "test.pkl")

        aggregated_dir: str = os.path.join(data_dir, "aggregated")
        train_data_agg: str = os.path.join(aggregated_dir, "train.pkl")
        val_data_agg: str = os.path.join(aggregated_dir, "validation.pkl")
        test_data_agg: str = os.path.join(aggregated_dir, "test.pkl")
        
    dataset: Dataset = Dataset()