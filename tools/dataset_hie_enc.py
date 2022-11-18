import os
from utils.dataloader import create_one_hierarchy_data, pascal_tree
import argparse


parser = argparse.ArgumentParser(description='Encode Hie Data')
parser.add_argument("--gtdatadir", type=str, default="./data/pascal_part/gt_masks")
parser.add_argument("--save_dir", type=str, default="./data/pascal_part/gt_masks_hie")

args = parser.parse_args()
create_one_hierarchy_data(args.gtdatadir, args.save_dir,  pascal_tree)
