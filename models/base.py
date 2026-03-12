import numpy as np
import torch
from torch import nn

from utils.toolkit import accuracy


class BaseLearner:
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self.topk = 5
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        return self._network.feature_dim

    def after_task(self):
        pass

    def incremental_train(self):
        pass

    def _evaluate(self, y_pred, y_true):
        grouped = accuracy(
            y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"]
        )
        return {
            "grouped": grouped,
            "top1": grouped["total"],
            f"top{self.topk}": np.around(
                (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true), decimals=2
            ),
        }

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        return self._evaluate(y_pred, y_true), None

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
