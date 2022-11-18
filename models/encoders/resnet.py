from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torch.utils.model_zoo import load_url
import os


class ResNetBackbone(ResNet):
    def __init__(self, out_channels: tuple, name: str, **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = 5
        self.name = name
        self.out_channels = out_channels

        del self.fc
        del self.avgpool

    def forward(self, x: Tensor):
        output = []
        # output.append(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        output.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        output.append(x)

        x = self.layer2(x)
        output.append(x)

        x = self.layer3(x)
        output.append(x)

        x = self.layer4(x)
        output.append(x)

        return output

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
                        strict: bool = True):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict)


resnet_backbones = {"resnet18": {
    "weights_url": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",
    "out_channels": (3, 64, 64, 128, 256, 512),
    "params": {
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
    },
},
    "resnet34": {
        "weights_url": "",
        "out_channels": (3, 64, 256, 512, 1024, 2048),
        "params": {
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "weights_url": "",
        "out_channels": (3, 64, 256, 512, 1024, 2048),
        "params": {
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "weights_url": "",
        "out_channels": (3, 64, 256, 512, 1024, 2048),
        "params": {
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "weights_url": "",
        "out_channels": (3, 64, 256, 512, 1024, 2048),
        "params": {
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
}


def init_resnet(resnet_name: str = 'resnet18',
                pretrained: bool = False):
    params = resnet_backbones[resnet_name]["params"]
    model = ResNetBackbone(resnet_backbones[resnet_name]["out_channels"],
                           resnet_name,
                           **params)
    if pretrained:

        if not os.path.exists("./pretrained_models"):
            os.mkdir("./pretrained_models")

        try:
            state_dict = load_url(resnet_backbones[resnet_name]["weights_url"], model_dir="./pretrained_models")
        except:
            raise ValueError("Failed To Load Model")
        model.load_state_dict(state_dict)
    return model
