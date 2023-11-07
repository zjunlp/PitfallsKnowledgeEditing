import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

class ConflictEditDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        with open(data_dir) as fp:
            self.data = json.load(fp)
        if size is not None:
            self.data = self.data[:size]
        
        print(f"Loaded ConflictEdit with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
