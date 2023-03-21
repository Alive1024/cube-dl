from torchvision import transforms

from wrappers import BasicDataWrapper
from datasets.oracle_mnist import OracleMNISTMemoryDataset


def get_data_wrapper_instance() -> BasicDataWrapper:
    data_dir = "data/Oracle-MNIST"
    return BasicDataWrapper(
        batch_size=64,
        dataset_fit=OracleMNISTMemoryDataset(data_dir=data_dir,
                                             split="train",
                                             transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,), (0.5,))
                                              ])),
        dataset_test=OracleMNISTMemoryDataset(data_dir=data_dir,
                                              split="t10k",
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5,), (0.5,))
                                              ])),
        dataloader_num_workers=0,
        auto_split_train_val=0.9
    )
