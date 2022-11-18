import argparse
import json
from utils.dataloader import PascalPartHierarchyDataset, PascalTestTransform
from models.hseg_model import HSEG
from torch.utils.data import DataLoader
from configs.config import config
from manage.evaluation_manager import EvalManager


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

    val_dataset = PascalPartHierarchyDataset(root_dir=config_exp["root_data_dir"],
                                             img_dir=config_exp["img_dir"],
                                             gt_dir=config_exp["gt_dir"],
                                             sample_ids_file=config_exp["val_sample_ids_file"],
                                             transform=PascalTestTransform,
                                             )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    hseg = HSEG(config_exp)
    eval_manager = EvalManager(config_exp,
                               hseg,
                               val_loader)
    eval_manager.evaluate()


if __name__ == '__main__':
    main()

