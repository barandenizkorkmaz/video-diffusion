import os.path

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np

from src.utils.dataset import get_splitting_dates

class ASIDataset(Dataset):

    def __init__(self, timestamps, asi_reader, transform=None):
        self.timestamps = timestamps
        self.asi_reader = asi_reader
        self.transform = transform

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        timestamp = self.timestamps[idx]
        asi = self.asi_reader(timestamp)
        height, width = asi.shape[-2:]
        asi = asi.reshape(-1, height, width).transpose((1, 2, 0))
        if self.transform:
            asi = self.transform(asi)
        len_target = len(self.asi_reader.post_asi) * 3
        previous = asi[:-len_target]
        target = asi[-len_target:]
        return previous, target

class ASIDatamodule(pl.LightningDataModule):

    def __init__(
            self,
            asi_reader,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
            train_transform=None,
            eval_transform=None,
            split_dir=None
    ):
        super().__init__()
        timestamps = asi_reader.get_valid_timestamps(asi_reader.all_timestamps)

        train_dates, valid_dates, test_dates = get_splitting_dates(
            train=os.path.join(split_dir, 'train_dates.csv'),
            valid=os.path.join(split_dir, 'valid_dates.csv'),
            test=os.path.join(split_dir, 'test_dates.csv')
        )

        # set the train, validation and test timestamps as the timestamps that correspond to a date in the respective set
        self.train_ts = timestamps[pd.DatetimeIndex(timestamps.date).isin(train_dates)]
        self.val_ts = timestamps[pd.DatetimeIndex(timestamps.date).isin(valid_dates)]
        self.val_ts = np.random.RandomState(seed=0).permutation(val_ts) ???
        self.test_ts = timestamps[pd.DatetimeIndex(timestamps.date).isin(test_dates)]

        self.asi_reader = asi_reader
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ASIDataset(self.train_ts, self.asi_reader, self.train_transform)
            self.val_dataset = ASIDataset(self.val_ts, self.asi_reader, self.eval_transform)
        if stage == 'test':
            self.test_dataset = ASIDataset(self.test_ts, self.asi_reader, self.eval_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size,
                          num_workers=1, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=False)