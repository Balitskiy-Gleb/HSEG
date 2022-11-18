import argparse
import json
from manage.train_manager import TrainManager
from utils.dataloader import PascalPartHierarchyDataset, PascalTestTransform
from models.hseg_model import HSEG
from torch.utils.data import DataLoader
from utils.loss import LossFunc, init_optimizer
from configs.config import config


def main():
    parser = argparse.ArgumentParser(description='Train HSEG System')
    parser.add_argument("--cfg", type=str, default=None)
    args = parser.parse_args()
    if args.cfg is not None:
        with open(args.cfg, "r") as fp:
            config_exp = json.load(fp)
    else:
        config_exp = config.copy()
    print(config_exp)
    train_dataset = PascalPartHierarchyDataset(root_dir=config_exp["root_data_dir"],
                                               img_dir=config_exp["img_dir"],
                                               gt_dir=config_exp["gt_dir"],
                                               sample_ids_file=config_exp["train_sample_ids_file"],
                                               )
    val_dataset = PascalPartHierarchyDataset(root_dir=config_exp["root_data_dir"],
                                             img_dir=config_exp["img_dir"],
                                             gt_dir=config_exp["gt_dir"],
                                             sample_ids_file=config_exp["val_sample_ids_file"],
                                             transform = PascalTestTransform,
                                             )
    train_loader = DataLoader(train_dataset, batch_size=config_exp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config_exp["batch_size"], shuffle=True)
    hseg = HSEG(config_exp)
    optimizer, scheduler = init_optimizer(hseg, config_exp)
    loss = LossFunc(config_exp)
    train_manager = TrainManager(config_exp,
                                 hseg,
                                 train_loader,
                                 val_loader,
                                 optimizer,
                                 scheduler,
                                 loss
                                 )
    train_manager.train()





if __name__ == '__main__':
    main()
