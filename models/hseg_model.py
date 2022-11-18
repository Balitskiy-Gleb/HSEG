import torch
import torch.nn as nn
from models.encoders.resnet import ResNetBackbone, init_resnet
from models.decoders.Unet import UnetDecoder
from typing import Any, Callable, List, Optional, Type, Union, Dict

# if finetune:
#     if os.path.exists(checkpoint):
#         state_dict = torch.load(checkpoint)
#         model.load_state_dict(state_dict)
#     else:
#         raise FileNotFoundError(checkpoint + " Not Found")



class HSEG(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.encoder = init_resnet(resnet_name=config["encoder"],
                                   pretrained=config["pretrained"])

        self.n_blocks = self.encoder.n_blocks
        self.out_channels = self.encoder.out_channels[::-1]
        self.decoder = UnetDecoder(in_channels=self.out_channels[0],
                                   out_channels=self.config["n_classes"],
                                   skip_channels=self.out_channels[1:],
                                   n_blocks=self.n_blocks)

        self.seg_head = None
        self.class_head = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.seg_head is not None:
            self.seg_head(x)
        if self.class_head is not None:
            x = self.class_head(x)
        return x






