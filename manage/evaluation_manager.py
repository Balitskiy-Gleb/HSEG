from models.hseg_model import HSEG
from torch.utils.data import DataLoader
from utils.metrics import pascal_root_leaf_paths
from typing import Dict
import torch
from utils.metrics import ScoreMeter
import os
from tqdm import tqdm
import torchvision.transforms as T


class EvalManager:
    def __init__(self,
                 config: Dict,
                 hseg: HSEG,
                 test_loader: DataLoader):
        self.config = config
        self.hseg = hseg
        self.val_loader = test_loader
        self.save_dir_paths = []
        self._init_experiment()
        self.device = config["device"]
        self._on_device()
        self.score_meter = ScoreMeter(self.config)

        self.toPIL = T.ToPILImage()
        self.palette = None
        self.colors = {}
        self.set_up_colors()
        self.cur_batch = 0
        self.load_checkpoint()

    def load_checkpoint(self):

        filename = os.path.join(self.config["root_dir"], self.config["eval_checkpoint"])
        self.hseg.load_state_dict(torch.load(filename))
        print("Check Point Loaded: ", filename)

    def set_up_colors(self):
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.colors[1] = torch.as_tensor([i for i in range(7)])[:, None] * self.palette
        self.colors[1] = (self.colors[1] % 255).numpy().astype("uint8")

        self.colors[2] = torch.as_tensor([i for i in range(3)])[:, None] * self.palette
        self.colors[2] = (self.colors[2] % 255).numpy().astype("uint8")

        self.colors[3] = torch.as_tensor([i for i in range(2)])[:, None] * self.palette
        self.colors[3] = (self.colors[3] % 255).numpy().astype("uint8")

    def _init_experiment(self):
        if not os.path.exists(os.path.join(self.config["root_dir"], "experiments")):
            os.mkdir(os.path.join(self.config["root_dir"], "experiments"))

        self.save_dir = os.path.join(self.config["root_dir"], "experiments",
                                     self.config["exp_name"])
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        dir_names = ["pictures"]

        for name in dir_names:
            self.save_dir_paths.append(os.path.join(self.save_dir, name))
            if not os.path.exists(self.save_dir_paths[-1]):
                os.mkdir(self.save_dir_paths[-1])

    def _on_device(self):
        self.hseg.to(self.device)

    def save_preds(self, prediction, gt, scores):
        gt = gt.to('cpu').type(torch.uint8)
        prediction = prediction.to('cpu')
        predictions_eval = torch.zeros_like(prediction[:, 0:7, :, :])
        for key in pascal_root_leaf_paths.keys():
            predictions_eval[:, key, :, :] = prediction[:, [key] + pascal_root_leaf_paths[key], :, :].sum(1)
        predictions_eval = predictions_eval.max(1)[1]
        gt_class = gt[:, 0:7, :, :].max(1)[1]
        self.save_level_masks(predictions_eval, 1, "mIoU1_img", scores["mIoU1"])
        self.save_level_masks(gt_class, 1, "mIoU1_mask", scores["mIoU1"])

        for key in pascal_root_leaf_paths.keys():
            if key == 0:
                continue
            predictions_eval[predictions_eval == key] = pascal_root_leaf_paths[key][0] - 7
        gt_class = gt[:, [0, 8, 9], :, :].max(1)[1]
        self.save_level_masks(predictions_eval, 2, "mIoU2_img", scores["mIoU2"])
        self.save_level_masks(gt_class, 2, "mIoU2_mask", scores["mIoU2"])

        predictions_eval[predictions_eval != 0] = 1
        gt_class = gt[:, [0, 7], :, :].max(1)[1]
        self.save_level_masks(predictions_eval, 3, "mIoU3_img", scores["mIoU3"])
        self.save_level_masks(gt_class, 3, "mIoU3_mask", scores["mIoU3"])

    def save_level_masks(self, prediction, level, name_score, score):
        pred_img = self.toPIL(prediction[0].type(torch.uint8))
        pred_img.putpalette(self.colors[level])
        pic_name = "pred_" + name_score + "_" + str(self.cur_batch)+ ".png"
        save_path = os.path.join(self.save_dir_paths[0], pic_name)
        pred_img.save(save_path)

    def evaluate(self):
        loss_v = 0
        self.cur_batch = 0
        self.hseg.eval()
        with torch.no_grad():
            for img, gt in tqdm(self.val_loader, desc="Test Batch:"):
                self.cur_batch += 1
                img, gt = img.to(self.device), gt.to(self.device)
                prediction = self.hseg(img)
                # loss_v += self.loss(prediction, gt)
                scores = self.score_meter.update(prediction, gt)
                self.save_preds(prediction, gt, scores)
        print("Scores : ", self.score_meter.get_mean())
