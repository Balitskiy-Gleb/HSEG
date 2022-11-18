import json
import os
from torch.cuda import is_available
config = {}
config["exp_name"] = "hseg_bce_100"
config["encoder"] = 'resnet18'
config["pretrained"] = True
config["n_classes"] = 10
config["device"] = 'cuda:0' if is_available() else 'cpu'
#config["device"] = 'cpu'
## Dirs configuration
abs_path = os.path.abspath("../HSEG_test")
config["root_dir"] = abs_path
config["root_data_dir"] = "data/pascal_part/"
config["img_dir"] = "JPEGImages"
config["gt_dir"] = "gt_masks_hie"
config["train_sample_ids_file"] = "train_id.txt"
config["val_sample_ids_file"] = "val_id.txt"

## Train Params
config["optimizer"] = "sgd"
config["lr"] = 0.01
config["batch_size"] = 15
config["metrics"] = "mIoU"
config["criterion"] = 'BCE'
config["gamma"] = 0
config["n_epochs"] = 100
config["save_epoch"] = 10
##
config["eval_checkpoint"] = os.path.join("./experiments/", config["exp_name"], "model_checkpoints", "checkpoint_99.pt") 
path = os.path.join(config["root_dir"], "configs", config["exp_name"] + ".json")

with open(path, "w") as file_json:
    json.dump(config, file_json, indent=2)
