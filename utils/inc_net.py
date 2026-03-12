import logging
from pathlib import Path

import torch
from torch import nn
import timm


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _load_local_vit_weights(model_name, num_classes=0):
    candidates = {
        "vit_base_patch16_224": [
            _repo_root() / "pretrained" / "vit_base_patch16_224_augreg_in21k_ft_in1k.bin",
            _repo_root() / "pretrained" / "pytorch_model.bin",
        ],
    }
    for candidate in candidates.get(model_name, []):
        if not candidate.exists():
            continue
        logging.info("Loading local pretrained weights from %s", candidate)
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        state_dict = torch.load(candidate, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict):
            state_dict = {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
                if not key.startswith("head.")
            }
        model.load_state_dict(state_dict, strict=False)
        return model
    return None


def _create_vit_base_patch16_224(num_classes=0):
    local_model = _load_local_vit_weights("vit_base_patch16_224", num_classes=num_classes)
    if local_model is not None:
        return local_model
    logging.info("Local pretrained weights not found, falling back to timm download.")
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)


def get_backbone(args, pretrained=False):
    del pretrained
    name = args["backbone_type"].lower()
    if name in {"pretrained_vit_b16_224", "vit_base_patch16_224"}:
        model = _create_vit_base_patch16_224(num_classes=0)
    elif name in {"pretrained_vit_b16_224_in21k", "vit_base_patch16_224_in21k"}:
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    else:
        raise NotImplementedError(
            f"Unknown backbone type {name}. This repository only keeps the CoSO ViT backbones."
        )

    model.out_dim = 768
    return model.eval()


class CoSOVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)

        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList(
            [nn.Linear(args["embd_dim"], self.class_num, bias=True) for _ in range(args["total_sessions"])]
        )
        self.numtask = 0

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes):
        del nb_classes
        self.numtask += 1

    def forward(self, x):
        image_features = self.backbone(x)

        if self.training:
            logits = self.classifier_pool[self.numtask - 1](image_features)
        else:
            logits = []
            for classifier in self.classifier_pool[: self.numtask]:
                logits.append(classifier(image_features))
            logits = torch.cat(logits, dim=1)

        return {"features": image_features, "logits": logits}
