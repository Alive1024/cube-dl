from cube.config_sys import cube_data_module, shared_config
from torchvision import transforms as F  # noqa: N812
from torchvision.datasets import MNIST

from datasets.basic_data_module import BasicDataModule


@cube_data_module
def get_mnist_data_module():
    return BasicDataModule(
        default_batch_size=shared_config.get("batch_size"),
        dataset_fit=MNIST(
            root="data",
            train=True,
            transform=F.Compose([F.ToTensor(), F.Normalize((0.1307,), (0.3081,))]),
        ),
        dataset_val=MNIST(
            root="data",
            train=False,
            transform=F.Compose([F.ToTensor(), F.Normalize((0.1307,), (0.3081,))]),
        ),
        dataloader_num_workers=4,
    )
