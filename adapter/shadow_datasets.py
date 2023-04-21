import os.path
import torch
import cv2
import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional
import kornia
from pathlib import Path
import kornia.augmentation as K
from adapter import shadow_map_transforms

class ShadowTrainDataset(data.Dataset):
    def __init__(self, img_length, img_list_a, img_list_b, transform_config):
        self.img_length = img_length
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b
        self.transform_config = transform_config

        self.shadow_op = shadow_map_transforms.ShadowMapTransforms()
        # self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        file_name = self.img_list_a[idx].split("/")[-1].split(".png")[0]
        try:
            rgb_ws = cv2.imread(self.img_list_a[idx])
            rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
            state = torch.get_rng_state()
            rgb_ws = self.initial_op(rgb_ws)

            torch.set_rng_state(state)
            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.initial_op(rgb_ns)

            rgb_ws, rgb_ns, shadow_map, shadow_matte = self.shadow_op.generate_shadow_map(rgb_ws, rgb_ns, False)

            # rgb_ws_gray = kornia.color.rgb_to_grayscale(rgb_ws)
            # rgb_ws = self.norm_op(rgb_ws)
            # rgb_ws_gray = self.norm_op(rgb_ws_gray)
            # rgb_ns = self.norm_op(rgb_ns)
            # shadow_map = self.norm_op(shadow_map)
            # shadow_matte = self.norm_op(shadow_matte)

        except Exception as e:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            print("ERROR: ", e)
            rgb_ws = None
            rgb_ws_gray = None
            rgb_ns = None
            shadow_map = None
            shadow_matte = None

        return file_name, rgb_ws, rgb_ns, shadow_map, shadow_matte

    def __len__(self):
        return self.img_length

class ShadowISTDDataset(data.Dataset):
    def __init__(self, img_length, img_list_a, img_list_b, img_list_c, transform_config):
        self.img_length = img_length
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b
        self.img_list_c = img_list_c
        self.transform_config = transform_config

        self.shadow_op = shadow_map_transforms.ShadowMapTransforms()
        # self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 320)),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        file_name = self.img_list_a[idx].split("/")[-1].split(".png")[0]

        try:
            rgb_ws = cv2.imread(self.img_list_a[idx])
            rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
            rgb_ws = self.initial_op(rgb_ws)

            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.initial_op(rgb_ns)

            # shadow_mask = cv2.imread(self.img_list_c[idx])
            # shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
            # shadow_mask = self.initial_op(shadow_mask)

            shadow_mask = rgb_ns - rgb_ws
            shadow_mask = kornia.color.rgb_to_grayscale(shadow_mask)
            # shadow_mask = self.initial_op(shadow_mask)

        except Exception as e:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            print("ERROR: ", e)
            rgb_ws = None
            rgb_ns = None
            rgb_ws_gray = None
            shadow_mask = None
            shadow_matte = None

        return file_name, rgb_ws, rgb_ns, shadow_mask

    def __len__(self):
        return self.img_length

class ShadowSRDDataset(data.Dataset):
    def __init__(self, img_length, img_list_a, img_list_b, img_list_c, transform_config):
        self.img_length = img_length
        self.img_list_a = img_list_a
        self.img_list_b = img_list_b
        self.img_list_c = img_list_c
        self.transform_config = transform_config

        self.shadow_op = shadow_map_transforms.ShadowMapTransforms()
        # self.norm_op = transforms.Normalize((0.5, ), (0.5, ))

        self.initial_op = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(constants.TEST_IMAGE_SIZE),
            # transforms.Resize((160, 210)),
            transforms.Resize((240, 320)),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        file_name = self.img_list_a[idx].split("/")[-1].split(".")[0]

        try:
            rgb_ws = cv2.imread(self.img_list_a[idx])
            rgb_ws = cv2.cvtColor(rgb_ws, cv2.COLOR_BGR2RGB)
            rgb_ws = self.initial_op(rgb_ws)

            rgb_ns = cv2.imread(self.img_list_b[idx])
            rgb_ns = cv2.cvtColor(rgb_ns, cv2.COLOR_BGR2RGB)
            rgb_ns = self.initial_op(rgb_ns)

            # shadow_mask = cv2.imread(self.img_list_c[idx])
            # shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)
            # shadow_mask = self.initial_op(shadow_mask)

            shadow_mask = rgb_ns - rgb_ws
            shadow_mask = kornia.color.rgb_to_grayscale(shadow_mask)
            # shadow_mask = self.initial_op(shadow_mask)

        except Exception as e:
            print("Failed to load: ", self.img_list_a[idx], self.img_list_b[idx])
            print("ERROR: ", e)
            rgb_ws = None
            rgb_ns = None
            rgb_ws_gray = None
            shadow_mask = None
            shadow_matte = None

        return file_name, rgb_ws, rgb_ns, shadow_mask

    def __len__(self):
        return self.img_length
