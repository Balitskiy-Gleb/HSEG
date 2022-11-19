import torch
from torch import Tensor
from typing import Dict
from torchmetrics.classification import MulticlassJaccardIndex

pascal_root_leaf_paths = {0: [],
                          1: [8, 7],
                          2: [8, 7],
                          3: [9, 7],
                          4: [8, 7],
                          5: [9, 7],
                          6: [8, 7]
                          }


class ScoreMeter:
    def __init__(self, config: Dict):
        self.config = config
        self.num_samples = 0
        if config['metrics'] == "mIoU":
            self.int_un = {"mIoU1": torch.zeros((7, 2)),
                           "mIoU2": torch.zeros((3, 2)),
                           "mIoU3": torch.zeros((2, 2))}

            self.mean_scores = {"mIoU1": 0,
                                "mIoU2": 0,
                                "mIoU3": 0}
            self.score_func = mIOU
        else:
            raise NotImplementedError("Choose mIoU")

    def update(self, prediction: Tensor, gt: Tensor):
        scores = self.score_func(prediction, gt, self.config["device"])
        for key in scores.keys():
            self.int_un[key] += scores[key]
            self.num_samples += 1
            self.mean_scores[key] = (self.int_un[key][1:, 0] / self.int_un[key][1:, 1]).mean()
        return scores

    def clear(self):
        self.num_samples = 0
        self.int_un = {"mIoU1": torch.zeros((7, 2)),
                       "mIoU2": torch.zeros((3, 2)),
                       "mIoU3": torch.zeros((2, 2))}
        self.mean_scores = {"mIoU1": 0, "mIoU2": 0, "mIoU3": 0}

    def get_mean(self):
        return self.mean_scores


def iou_custom(preds, target, num_classes=7, device='cpu'):
    iou = torch.zeros((num_classes, 2))
    iou = iou.to(device)
    for cl in range(num_classes):
        preds_cl = (preds == cl)
        target_cl = (target == cl)
        inter = torch.logical_and(preds_cl, target_cl)
        union = torch.logical_or(preds_cl, target_cl)
        iou[cl, 0] = inter.sum()
        iou[cl, 1] = union.sum()
    return iou


def mIOU(prediction: Tensor, gt: Tensor, device='cpu'):

    predictions_eval = torch.zeros_like(prediction[:, 0:7, :, :]).to(device)
    for key in pascal_root_leaf_paths.keys():
        predictions_eval[:, key, :, :] = prediction[:, [key] + pascal_root_leaf_paths[key], :, :].sum(1)
    predictions_eval = predictions_eval.max(1)[1]
    gt_class = gt[:, 0:7, :, :].max(1)[1]
    iou_1_score = iou_custom(predictions_eval, gt_class, 7, device)

    for key in pascal_root_leaf_paths.keys():
        if key == 0:
            continue
        predictions_eval[predictions_eval == key] = pascal_root_leaf_paths[key][0] - 7

    gt_class = gt[:, [0, 8, 9], :, :].max(1)[1]
    iou_2_score = iou_custom(predictions_eval, gt_class, 3, device)

    predictions_eval[predictions_eval != 0] = 1
    gt_class = gt[:, [0, 7], :, :].max(1)[1]
    iou_3_score = iou_custom(predictions_eval, gt_class, 2, device)

    mean_scores = {"mIoU1": iou_1_score.to('cpu'),
                   "mIoU2": iou_2_score.to('cpu'),
                   "mIoU3": iou_3_score.to('cpu')}
    return mean_scores
