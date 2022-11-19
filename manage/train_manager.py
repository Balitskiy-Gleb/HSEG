from models.hseg_model import HSEG
from torch.utils.data import DataLoader
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import torch
from utils.loss import LossFunc
from utils.metrics import ScoreMeter
import os
import json

from tqdm import tqdm

class TrainManager:
    def __init__(self,
                 config: Dict,
                 hseg: HSEG,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer,
                 scheduler,
                 loss: LossFunc,
                 ):
        self.hseg = hseg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.config = config
        self.score_meter = ScoreMeter(self.config)
        self.save_dir = None
        self.save_dir_paths = []
        self._init_experiment()
        self.device = config["device"]
        self._on_device()
        self.batch_stats = {"loss": [],
                            "lr": []}

        self.epoch_stats = {"loss_train": [],
                            "loss_val": [],
                            "metrics": [],
                            "lr": [],
                            }
        self.cur_epoch = 0
        self.cur_batch = 0
        self.log_path_tens = os.path.join(self.config["root_dir"], "tensorboard")

        if not os.path.exists(self.log_path_tens):
            os.mkdir(self.log_path_tens)
        self.log_path_tens = os.path.join(self.log_path_tens, config["exp_name"])
        if not os.path.exists(self.log_path_tens):
            os.mkdir(self.log_path_tens)
        self.writer = SummaryWriter(self.log_path_tens)


    def _init_experiment(self):
        if not os.path.exists(os.path.join(self.config["root_dir"], "experiments")):
            os.mkdir(os.path.join(self.config["root_dir"], "experiments"))

        self.save_dir = os.path.join(self.config["root_dir"], "experiments",
                                     self.config["exp_name"])
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        dir_names = ["model_checkpoints", "tensorboard", "configs"]

        for name in dir_names:
            self.save_dir_paths.append(os.path.join(self.save_dir, name))
            if not os.path.exists(self.save_dir_paths[-1]):
                os.mkdir(self.save_dir_paths[-1])
        with open(os.path.join(self.save_dir_paths[-1], "config.json"), "w") as file:
            json.dump(self.config, file)

    def _on_device(self):
        self.hseg.to(self.device)

    def register_train_batch(self, loss_t):
        self.batch_stats["loss"].append(loss_t)
        self.batch_stats["lr"].append(self.scheduler.get_last_lr()[0])
        self.writer.add_scalar("loss", loss_t, self.cur_batch)
        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.cur_batch)

    def register_epoch_value(self, value, name):
        print(name, " : " , value)
        self.epoch_stats[name].append(value)
        if name == 'metrics':
            if self.config['metrics'] == 'mIoU':
                for key in value.keys():
                    self.writer.add_scalar(key, value[key], self.cur_epoch)
        else:
            self.writer.add_scalar(name, value, self.cur_epoch)

    def train_one_epoch(self):
        accum_loss = 0
        len_loader = len(self.train_loader)
        for img, gt in tqdm(self.train_loader, desc="Batch:"):
            self.optimizer.zero_grad()
            self.cur_batch += 1
            img, gt = img.to(self.device),  gt.to(self.device)
            prediction = self.hseg(img)
            loss_t = self.loss(prediction, gt)
            loss_t.backward()
            self.optimizer.step()
            self.scheduler.step()

            accum_loss += loss_t.detach().cpu().item()
            self.register_train_batch(loss_t.detach().cpu().item())
            
        self.register_epoch_value(accum_loss / len_loader, "loss_train")
        self.register_epoch_value(self.scheduler.get_last_lr()[0], "lr")

    def validate(self):
        loss_v = 0
        with torch.no_grad():
            for img, gt in tqdm(self.val_loader, desc="Val Batch:"):
                img, gt = img.to(self.device), gt.to(self.device)
                prediction = self.hseg(img)
                loss_v += self.loss(prediction, gt)
                scores = self.score_meter.update(prediction, gt)
            #print(self.hseg(img)[0,:,120:125,100])
            #print(gt[0,:,120:125,100])
            
        self.register_epoch_value(self.score_meter.get_mean(), "metrics")
        self.register_epoch_value(loss_v / len(self.val_loader), "loss_val")

    def finish_epoch(self):
        if self.cur_epoch % self.config["save_epoch"] == 0 or (self.cur_epoch + 1) == self.config["n_epochs"]:
            checkpoint_save_path = os.path.join(self.save_dir_paths[0],
                                                "checkpoint_" + str(self.cur_epoch) + ".pt")
            optimizer_save_path = os.path.join(self.save_dir_paths[0],
                                               "optimizer" + str(self.cur_epoch) + ".pt")

            torch.save(self.hseg.state_dict(), checkpoint_save_path)
            torch.save(self.optimizer.state_dict(), optimizer_save_path)
            self.score_meter.clear()
    def train(self):
        for epoch in tqdm(range(self.config["n_epochs"]), desc="Epoch:"):
            self.cur_epoch = epoch
            self.hseg.train()
            self.train_one_epoch()
            self.hseg.eval()
            self.validate()
            self.finish_epoch()
        self.writer.close()
