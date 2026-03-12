import numpy as np


def count_parameters(model, trainable=False):
    if trainable:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {"total": np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)}

    idxes = np.where(np.logical_and(y_true >= 0, y_true < init_cls))[0]
    label = f"{str(0).rjust(2, '0')}-{str(init_cls - 1).rjust(2, '0')}"
    all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = f"{str(class_id).rjust(2, '0')}-{str(class_id + increment - 1).rjust(2, '0')}"
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    return all_acc


def split_images_labels(imgs):
    images, labels = [], []
    for image_path, label in imgs:
        images.append(image_path)
        labels.append(label)
    return np.array(images), np.array(labels)
