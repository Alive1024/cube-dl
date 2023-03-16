import os
import os.path as osp
import gzip
from typing import Literal, Optional, Callable

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class OracleMNISTMemoryDataset(Dataset):
    def __init__(self, data_dir, split: Literal["train", "t10k"] = "train",
                 transform: Optional[Compose] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        super(OracleMNISTMemoryDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform

        labels_path = os.path.join(data_dir, '%s-labels-idx1-ubyte.gz' % split)
        images_path = os.path.join(data_dir, '%s-images-idx3-ubyte.gz' % split)

        with gzip.open(labels_path, 'rb') as lbpath:
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape((len(self.labels), 28, 28))

        print('The size of %s set: %d' % (split, len(self.labels)))

    def __getitem__(self, index):
        img, label = self.images[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.images)


class OracleMNISTExtractedDataset(Dataset):
    def __init__(self, data_dir, split: Literal["train", "test"] = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        split_path = osp.join(data_dir, split)
        self.image_paths = []
        self.labels = []
        for filename in os.listdir(split_path):
            self.image_paths.append(osp.join(split_path, filename))
            self.labels.append(int(osp.splitext(filename[filename.find('-')+1:])[0]))

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.image_paths)
