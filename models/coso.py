import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CoSOVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

from utils.scheduler import CosineSchedule
from coso.adamw import CoSOAdamW
import wandb

NUM_WORKERS = 16
COSO_TARGET_KEY = "attn.proj"


def _safe_wandb_log(payload):
    if getattr(wandb, "run", None) is not None:
        wandb.log(payload)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CoSOVitNet(args, True)
        logging.info("Use CoSO to train the network ...")

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _collect_coso_params(self):
        coso_params = []
        for module_name, module in self._network.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if COSO_TARGET_KEY not in module_name:
                continue
            coso_params.append(module.weight)
        return coso_params

    def _build_param_groups(self):
        coso_params = self._collect_coso_params()
        coso_param_ids = {id(param) for param in coso_params}
        regular_params = [
            param for param in self._network.parameters() if id(param) not in coso_param_ids
        ]

        return [
            {"params": regular_params, "lr": self.args["fc_lrate"]},
            {
                "params": coso_params,
                "lr": self.args["lrate"],
                "proj_rank": self.args["proj_rank"],
                "rank": self.args["rank"],
                "db_decay": self.args["db_decay"],
                "update_proj_gap": self.args["update_gap"],
            },
        ]

    def _configure_trainable_params(self):
        trainable_suffixes = ["attn.proj.weight", f"classifier_pool.{self._cur_task}"]

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if any(suffix in name for suffix in trainable_suffixes):
                param.requires_grad_(True)

        enabled = sorted(
            name for name, param in self._network.named_parameters() if param.requires_grad
        )
        logging.info(f"Parameters to be updated: {enabled}")

    def _reset_optimizer_state_for_new_task(self):
        self.optimizer.resetting_lr(self.args["lrate"], self.args["fc_lrate"])

        for param_group in self.optimizer.param_groups:
            if "rank" not in param_group:
                continue

            logging.info("resetting optimizer state for task %s", self._cur_task)
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                projector = state["projector"]
                projector.update_historical_space(self.args["threshold"])
                projector.update_projector(self._cur_task)
                state["step"] = 0
                state.pop("exp_avg")
                state.pop("exp_avg_sq")

        logging.info("resetting done for task %s", self._cur_task)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        param_groups = self._build_param_groups()
        self._configure_trainable_params()

        if self._cur_task == 0:
            self.optimizer = CoSOAdamW(
                param_groups,
                weight_decay=self.args["weight_decay"],
                betas=(0.9, 0.999),
            )
        else:
            self._reset_optimizer_state_for_new_task()

        scheduler = CosineSchedule(self.optimizer, self.args["epochs"])
        self.train_function(train_loader, test_loader, self.optimizer, scheduler)

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            one_epoch_steps = 0
            
            _safe_wandb_log(
                {f"task {self._cur_task} group 0 learning rate": optimizer.param_groups[0]["lr"]}
            )
            _safe_wandb_log(
                {f"task {self._cur_task} group 1 learning rate": optimizer.param_groups[1]["lr"]}
            )
            
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes
                
                logits = self._network(inputs)["logits"] / self.args["temperature"]
                loss = F.cross_entropy(logits, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                one_epoch_steps += 1

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = 0
            avg_loss = losses / len(train_loader)
            
            info = (
                "Task {}, Epoch {}/{}, Steps {} => Loss {:.3f}, "
                "Train_accy {:.2f}, Test_accy {:.2f}"
            ).format(
                self._cur_task, epoch + 1, self.args["epochs"], one_epoch_steps, avg_loss, train_acc, test_acc
            )
            
            _safe_wandb_log({f"Task {self._cur_task} train epoch loss": avg_loss})
            _safe_wandb_log({f"Task {self._cur_task} train acc": train_acc})
            _safe_wandb_log({f"Task {self._cur_task} test acc": test_acc})

            prog_bar.set_description(info)

        logging.info(info)
