import os
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule


class BasicDataWrapper(LightningDataModule):
    def __init__(self,
                 *,  # Compulsory keyword arguments, for better readability in config files.
                 # Batch sizes
                 default_batch_size: int = 8,
                 fit_batch_size: Optional[int] = None,
                 val_batch_size: Optional[int] = None,
                 test_batch_size: Optional[int] = None,
                 predict_batch_size: Optional[int] = None,
                 # Dataset classes
                 dataset_fit: Optional[Dataset] = None,
                 dataset_val: Optional[Dataset] = None,
                 dataset_test: Optional[Dataset] = None,
                 dataset_predict: Optional[Dataset] = None,
                 # Other arguments
                 dataloader_num_workers: int = os.cpu_count(),
                 auto_split_train_val: Optional[Union[float, int]] = None,
                 dataloader_pin_memory: bool = False
                 ):
        """
        """
        super(BasicDataWrapper, self).__init__()

        self.default_batch_size = default_batch_size
        self.fit_batch_size = fit_batch_size if fit_batch_size else default_batch_size
        self.val_batch_size = val_batch_size if val_batch_size else default_batch_size
        self.test_batch_size = test_batch_size if test_batch_size else default_batch_size
        self.predict_batch_size = predict_batch_size if predict_batch_size else default_batch_size

        self.dataset_fit = dataset_fit
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.dataset_predict = dataset_predict

        self.dataloader_num_workers = dataloader_num_workers
        self.auto_split_train_val = auto_split_train_val
        self.dataloader_pin_memory = dataloader_pin_memory

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
        if self.dataset_fit is None:
            raise ValueError("The argument `dataset_fit` should be passed in to conduct fit.")
        return DataLoader(self.dataset_fit,
                          batch_size=self.fit_batch_size,
                          shuffle=True,
                          num_workers=self.dataloader_num_workers,
                          pin_memory=self.dataloader_pin_memory)

    def val_dataloader(self):
        if self.dataset_val is None:
            raise ValueError("The argument `dataset_val` should be passed in to conduct validation.")
        return DataLoader(self.dataset_val,
                          batch_size=self.val_batch_size,
                          num_workers=self.dataloader_num_workers,
                          pin_memory=self.dataloader_pin_memory)

    def test_dataloader(self):
        if self.dataset_test is None:
            raise ValueError("The argument `dataset_test` should be passed in to conduct test.")
        return DataLoader(self.dataset_test,
                          batch_size=self.test_batch_size,
                          num_workers=self.dataloader_num_workers,
                          pin_memory=self.dataloader_pin_memory)

    def predict_dataloader(self):
        if self.dataset_predict is None:
            raise ValueError("The argument `dataset_predict` should be passed in to conduct prediction.")
        return DataLoader(self.dataset_predict,
                          batch_size=self.predict_batch_size,
                          num_workers=self.dataloader_num_workers,
                          pin_memory=self.dataloader_pin_memory)
