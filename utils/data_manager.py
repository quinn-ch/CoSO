import logging

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data import iCIFAR100, iDomainNet, iImageNetR


class DataManager:
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."

        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode):
        if source == "train":
            data, targets = self._train_data, self._train_targets
        elif source == "test":
            data, targets = self._test_data, self._test_targets
        else:
            raise ValueError(f"Unknown data source {source}.")

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError(f"Unknown mode {mode}.")

        selected_data, selected_targets = [], []
        for idx in indices:
            class_mask = np.where(np.logical_and(targets >= idx, targets < idx + 1))[0]
            selected_data.append(data[class_mask])
            selected_targets.append(targets[class_mask])

        selected_data = np.concatenate(selected_data)
        selected_targets = np.concatenate(selected_targets)
        return DummyDataset(selected_data, selected_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        self._train_data = idata.train_data
        self._train_targets = idata.train_targets
        self._test_data = idata.test_data
        self._test_targets = idata.test_targets
        self.use_path = idata.use_path
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [idx for idx in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        return idx, image, self.labels[idx]


def _map_new_class_index(targets, order):
    return np.array(list(map(order.index, targets)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar100":
        return iCIFAR100()
    if name == "imagenetr":
        return iImageNetR(args)
    if name == "domain":
        return iDomainNet(args)
    raise NotImplementedError(f"Unknown dataset {dataset_name}.")


def pil_loader(path):
    with open(path, "rb") as file_obj:
        img = Image.open(file_obj)
        return img.convert("RGB")
