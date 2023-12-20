import os

import torch
from cube.core import CubeDataModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class BasicDataModule(LightningDataModule, CubeDataModule):
    def __init__(
        self,
        *,  # Compulsory keyword arguments, for better readability in config files.
        # Batch sizes
        default_batch_size: int = 8,
        fit_batch_size: int | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        predict_batch_size: int | None = None,
        # Dataset classes
        dataset_fit: Dataset | None = None,
        dataset_val: Dataset | None = None,
        dataset_test: Dataset | None = None,
        dataset_predict: Dataset | None = None,
        # Other arguments
        dataloader_num_workers: int = os.cpu_count(),
        auto_split_train_val: float | int | None = None,
        # Specific keyword arguments for dataloaders
        train_dataloader_kwargs=None,
        val_dataloader_kwargs=None,
        test_dataloader_kwargs=None,
        predict_dataloader_kwargs=None,
    ):
        super().__init__()

        if (not dataset_fit) and (not dataset_val) and (not dataset_test) and (not dataset_predict):
            raise ValueError("At least one dataset should be provided!")

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

        self.train_dataloader_kwargs = train_dataloader_kwargs
        self.val_dataloader_kwargs = val_dataloader_kwargs
        self.test_dataloader_kwargs = test_dataloader_kwargs
        self.predict_dataloader_kwargs = predict_dataloader_kwargs

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
            if isinstance(self.auto_split_train_val, float):
                train_ratio, val_ratio = self.auto_split_train_val, 1 - self.auto_split_train_val
                self.dataset_fit, self.dataset_val = random_split(
                    self.dataset_fit, [train_ratio, val_ratio], generator=torch.Generator().manual_seed(42)
                )
            else:  # int
                train_len, val_len = self.auto_split_train_val, original_train_set_len - self.auto_split_train_val
                train_ratio, val_ratio = train_len / original_train_set_len, val_len / original_train_set_len
                self.dataset_fit, self.dataset_val = random_split(
                    self.dataset_fit, [train_len, val_len], generator=torch.Generator().manual_seed(42)
                )

            print(
                f"Auto split into train & val sets from the original train set, "
                f"ratio: {train_ratio} : {val_ratio}, quantity: {len(self.dataset_fit)} : {len(self.dataset_val)}"
            )

    def train_dataloader(self) -> DataLoader:
        if self.dataset_fit is None:
            raise ValueError("The argument `dataset_fit` should be passed in to conduct fit.")

        if self.train_dataloader_kwargs is not None:
            return DataLoader(
                self.dataset_fit,
                batch_size=self.fit_batch_size,
                shuffle=True,
                num_workers=self.dataloader_num_workers,
                **self.train_dataloader_kwargs,
            )

        return DataLoader(
            self.dataset_fit,
            batch_size=self.fit_batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.dataset_val is None:
            raise ValueError("The argument `dataset_val` should be passed in to conduct validation.")

        if self.val_dataloader_kwargs is not None:
            return DataLoader(
                self.dataset_val,
                batch_size=self.val_batch_size,
                num_workers=self.dataloader_num_workers,
                **self.val_dataloader_kwargs,
            )

        return DataLoader(
            self.dataset_val,
            batch_size=self.val_batch_size,
            num_workers=self.dataloader_num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self.dataset_test is None:
            raise ValueError("The argument `dataset_test` should be passed in to conduct test.")

        if self.test_dataloader_kwargs is not None:
            return DataLoader(
                self.dataset_test,
                batch_size=self.test_batch_size,
                num_workers=self.dataloader_num_workers,
                **self.test_dataloader_kwargs,
            )

        return DataLoader(
            self.dataset_test,
            batch_size=self.test_batch_size,
            num_workers=self.dataloader_num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.dataset_predict is None:
            raise ValueError("The argument `dataset_predict` should be passed in to conduct prediction.")

        if self.predict_dataloader_kwargs is not None:
            return DataLoader(
                self.dataset_predict,
                batch_size=self.predict_batch_size,
                num_workers=self.dataloader_num_workers,
                **self.predict_dataloader_kwargs,
            )

        return DataLoader(
            self.dataset_predict,
            batch_size=self.predict_batch_size,
            num_workers=self.dataloader_num_workers,
        )
