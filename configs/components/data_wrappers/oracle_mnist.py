from torchvision import transforms

from wrappers import BasicDataWrapper
from config_decorators import data_wrapper_getter
from datasets.oracle_mnist import OracleMNISTMemoryDataset


@data_wrapper_getter
def get_data_wrapper_instance():
    data_dir = "data/Oracle-MNIST"
    test_predict_dataset = OracleMNISTMemoryDataset(data_dir=data_dir,
                                                    split="t10k",
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5,), (0.5,))
                                                    ]))
    return BasicDataWrapper(
        default_batch_size=64,
        dataset_fit=OracleMNISTMemoryDataset(data_dir=data_dir,
                                             split="train",
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))
                                             ])),
        dataset_test=test_predict_dataset,
        dataset_predict=test_predict_dataset,
        dataloader_num_workers=0,
        auto_split_train_val=0.9
    )
