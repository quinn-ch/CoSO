from pathlib import Path

import numpy as np
import yaml
from torchvision import datasets, transforms

from utils.toolkit import split_images_labels


class iData:
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR100(iData):
    use_path = False
    train_trsf = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    test_trsf = [transforms.Resize(224)]
    common_trsf = [transforms.ToTensor(), transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data = train_dataset.data
        self.train_targets = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_targets = np.array(test_dataset.targets)


class iImageNetR(iData):
    def __init__(self, args):
        del args
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        self.common_trsf = [transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        root = Path("./data/imagenet-r")
        if not (root / "train").is_dir() or not (root / "test").is_dir():
            raise FileNotFoundError("ImageNet-R dataset not found. Expected train/test under ./data/imagenet-r.")

        train_dset = datasets.ImageFolder(str(root / "train"))
        test_dset = datasets.ImageFolder(str(root / "test"))
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iDomainNet(iData):
    def __init__(self, args):
        del args
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        self.common_trsf = [transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]
        self.class_order = np.arange(345).tolist()

    def download_data(self):
        with open("utils/domainnet_trainb.yaml", "r") as train_file:
            train_data_config = yaml.load(train_file, Loader=yaml.Loader)
        with open("utils/domainnet_testb.yaml", "r") as test_file:
            test_data_config = yaml.load(test_file, Loader=yaml.Loader)

        self.train_data = np.array(train_data_config["data"])
        self.train_targets = np.array(train_data_config["targets"])
        self.test_data = np.array(test_data_config["data"])
        self.test_targets = np.array(test_data_config["targets"])
