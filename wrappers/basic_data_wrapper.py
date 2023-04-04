import os
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class BasicDataWrapper(LightningDataModule):
    def __init__(self, *,
                 batch_size: int,
                 dataset_fit: Dataset,
                 dataset_test: Dataset,
                 dataset_val: Optional[Dataset] = None,
                 dataset_predict: Optional[Dataset] = None,
                 dataloader_num_workers: int = os.cpu_count(),
                 auto_split_train_val: Optional[Union[float, int]] = None
                 ):
        """

        :param batch_size:
        :param dataset_fit:
        :param dataset_test:
        :param dataset_val:
        :param dataset_predict:
        :param auto_split_train_val:
        """
        super(BasicDataWrapper, self).__init__()

        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers

        self.dataset_fit = dataset_fit
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.dataset_predict = dataset_predict
        self.auto_split_train_val = auto_split_train_val

        # TODO 直接在文件上进行 predict

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # if stage == "fit":
        #     self.dataset_train = None
        # elif stage == "validate":
        #     pass
        # elif stage == "test":
        #     pass
        # elif stage == "predict":
        #     pass
        if (not self.dataset_val) and self.auto_split_train_val:
            original_train_set_len = len(self.dataset_fit)
            if type(self.auto_split_train_val) == float:
                train_ratio, val_ratio = self.auto_split_train_val, 1 - self.auto_split_train_val
                train_len, val_len = original_train_set_len * train_ratio, original_train_set_len * val_ratio
                self.dataset_fit, self.dataset_val = random_split(self.dataset_fit,
                                                                  [train_ratio, val_ratio],
                                                                  generator=torch.Generator().manual_seed(42))
            else:  # int
                train_len, val_len = self.auto_split_train_val, original_train_set_len - self.auto_split_train_val
                train_ratio, val_ratio = train_len / original_train_set_len, val_len / original_train_set_len
                self.dataset_fit, self.dataset_val = random_split(self.dataset_fit,
                                                                  [train_len, val_len],
                                                                  generator=torch.Generator().manual_seed(42))

            print(f"Auto split into train & val sets from the original train set, "
                  f"ratio: {train_ratio} : {val_ratio}, quantity: {len(self.dataset_fit)} : {len(self.dataset_val)}")

    def train_dataloader(self):
        return DataLoader(self.dataset_fit, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          num_workers=self.dataloader_num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=self.batch_size,
                          num_workers=self.dataloader_num_workers)
