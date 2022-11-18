import torch
from torch import Tensor
from typing import Dict
from utils.dataloader import pascal_tree_parents, pascal_tree_childs
from torch.optim import SGD, Adam
'''
├── (0) background
└── body(7)
    ├── upper_body (8)
    |   ├── (1) low_hand
    |   ├── (6) up_hand
    |   ├── (2) torso
    |   └── (4) head
    └── lower_body (9)
        ├── (3) low_leg
        └── (5) up_leg
'''

def init_optimizer(model, config):
    if config["optimizer"] == 'sgd':
        optimizer = SGD(model.parameters(),
                        lr=config["lr"],
                        momentum=0.9,
                        weight_decay=10 ** (-4))
    elif config["optimizer"] == 'adam':
        optimizer = Adam(model.parameters(),
                         lr=config["lr"],
                         betas=(0.9, 0.999), weight_decay=10 ** (-4))
    else:
        raise NotImplementedError("Not Implemented Optimizer (try SGD, ADAM) ")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=300,
                                                gamma=0.93,
                                                last_epoch=- 1)
    return optimizer, scheduler


class LossFunc(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.loss_name = config["criterion"]
        if self.loss_name not in ["BCE", "FocalLoss", "FocalTreeLoss"]:
            raise ValueError("Not implemented Loss")
        if "Focal" in self.loss_name:
            self.gamma = config["gamma"]
        else:
            self.gamma = None
        self.config = config

    def forward(self, prediction: Tensor, gt: Tensor):
        return eval("self."+self.loss_name + "(prediction, gt)")

    
    def compute_p_scores_indexes(self,scores, tree_parents, tree_childs):
        p_plus = torch.zeros(scores.shape).to(self.config["device"])
        p_minus = torch.zeros(scores.shape).to(self.config["device"])
        for node in tree_parents.keys():
            p_plus[:, node, :, :] = scores[:, [node] + tree_parents[node], :, :].min(1)[0]
        for node in tree_childs.keys():
            p_minus[:, node, :, :] = scores[:, [node] + tree_childs[node], :, :].max(1)[0]
        return p_plus, p_minus

    def BCE(self, prediction: Tensor, gt: Tensor):
        return torch.nn.functional.binary_cross_entropy(prediction, gt) * 10

    def FocalLoss(self, prediction: Tensor, gt: Tensor):
        EPS = 10**(-10)
        log_pred = torch.pow(1 - prediction, self.gamma) * torch.log(prediction + EPS)
        log_pred_inv = torch.pow(prediction, self.gamma) * torch.log(1 - prediction + EPS)
        return (- gt * log_pred - (1 - gt) * log_pred_inv).sum(1).mean()

    def FocalTreeLoss(self, prediction: Tensor, gt: Tensor):
        p_plus, p_minus = self.compute_p_scores_indexes(prediction,
                                                        pascal_tree_parents,
                                                        pascal_tree_childs)
        pv = gt * p_plus + (1 - gt) *  p_minus
        return self.FocalLoss(pv, gt)
