import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import Dict

# PascalTrainTransform = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                            transforms.RandomResizedCrop(size=480,
#                                                                         scale=(0.25, 1)
#                                                                         ),
#                                            ]
#                                           )
PascalTrainTransform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomCrop(size=480,
                                                                 padding=0,
                                                                 pad_if_needed=True),
                                           ]
                                          )
PascalTestTransform = transforms.Compose([transforms.CenterCrop(size=480)]
                                         )

pascal_tree = {
    1: [8, 7],
    2: [8, 7],
    3: [9, 7],
    4: [8, 7],
    5: [9, 7],
    6: [8, 7]
}

pascal_tree_parents = {0: [],
                       1: [8, 7],
                       2: [8, 7],
                       3: [9, 7],
                       4: [8, 7],
                       5: [9, 7],
                       6: [8, 7],
                       7: [],
                       8: [7],
                       9: [7]
                       }

pascal_tree_childs = {0: [],
                      1: [],
                      2: [],
                      3: [],
                      4: [],
                      5: [],
                      6: [],
                      7: [1, 2, 3, 4, 5, 6, 8, 9],
                      8: [1, 2, 4, 6],
                      9: [3, 5]
                      }


# def create_one_hot_tensor(gt: torch.Tensor, classes_tree=None):
#     if classes_tree is None:
#         classes_tree = pascal_tree
#     num_classes = len(classes_tree)
#     gt_one_hot = torch.eye(num_classes)[gt]
#     print(gt.shape)
#     for i in range(gt.shape[1]):
#         for j in range(gt.shape[2]):
#             positions = classes_tree[gt[0, i, j].item()]
#             gt_one_hot[0, i, j, positions] = 1
#             print(gt_one_hot[0, i, j, :])
#     return gt_one_hot.movedim(3, 1)

def create_one_hot_numpy_fast(gt, classes_tree=None, num_classes=10):
    if classes_tree is None:
        classes_tree = pascal_tree
    gt_one_hot = np.eye(num_classes)[gt]
    gt_copy = gt.copy()
    for i in pascal_tree.keys():
        if len(pascal_tree[i]) > 0:
            gt_copy = np.where(gt_copy == i, pascal_tree[i][0], gt_copy)
    gt_one_hot += np.eye(num_classes)[gt_copy]

    gt_copy = gt.copy()
    for i in pascal_tree.keys():
        if len(pascal_tree[i]) > 1:
            gt_copy = np.where(gt_copy == i, pascal_tree[i][1], gt_copy)
    gt_one_hot += np.eye(num_classes)[gt_copy]

    return gt_one_hot.astype(bool)


def save_gt_numpy(path, gt_one_hot, name):
    np.save(os.path.join(path, name), gt_one_hot)


def create_one_hierarchy_data(data_dir: str, data_save_dir: str, classes_tree: Dict = None):
    gt_filenames = os.listdir(data_dir)
    if not os.path.exists(data_save_dir):
        os.mkdir(data_save_dir)
    for i_file, file in enumerate(gt_filenames):
        print(f"{i_file}/{len(gt_filenames)}")
        gt = np.load(os.path.join(data_dir, file))
        gt_one_hot = create_one_hot_numpy_fast(gt, classes_tree)
        save_gt_numpy(data_save_dir, gt_one_hot, file)


class PascalPartHierarchyDataset(Dataset):
    def __init__(self, root_dir: str,
                 img_dir: str,
                 gt_dir: str,
                 transform=PascalTrainTransform,
                 sample_ids_file: str = None,
                 ):
        self.images_path = os.path.join(root_dir, img_dir)
        self.gt_path = os.path.join(root_dir, gt_dir)
        self.sample_ids_file = os.path.join(root_dir, sample_ids_file)
        with open(self.sample_ids_file, "r") as file:
            self.ids = [line.rstrip() for line in file]
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.ids[idx] + ".jpg")
        img = Image.open(image_path)
        img = self.to_tensor(img)

        mask_path = os.path.join(self.gt_path, self.ids[idx] + ".npy")
        mask = np.load(mask_path)
        mask = self.to_tensor(mask)

        img_mask = torch.cat([img[:, :, :], mask], dim=0)
        transformed_img_mask = self.transform(img_mask)
        img_tr = transformed_img_mask[:3, :, :]
        mask_tr = transformed_img_mask[3:, :, :]
        mask_tr_sum = mask_tr.sum(0) == 0.0
        mask_tr[0,:,:][mask_tr_sum] = 1.0
        return img_tr, mask_tr
