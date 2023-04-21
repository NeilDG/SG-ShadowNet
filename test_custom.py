import argparse

import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.utils

from utils.utils import weights_init_normal, tensor2img, calc_RMSE
from models.model import ConGenerator_S2F, ConRefineNet
from loss.losses import L_spa
from adapter import dataset_loader

def main():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    opts = {}
    opts["img_to_load"] = -1
    opts["num_workers"] = 6
    opts["cuda_device"] = "cuda:0"
    load_size = 8

    ###### Definition of variables ######
    # Networks
    netG_1 = ConGenerator_S2F()
    netG_2 = ConRefineNet()

    netG_1.cuda()
    netG_2.cuda()

    netG_1.apply(weights_init_normal)
    netG_2.apply(weights_init_normal)

    last_epoch = 20
    checkpoint_G_1 = torch.load('ckpt_fs/netG_1_%d.pth' % (last_epoch), map_location=device)
    checkpoint_G_2 = torch.load('ckpt_fs/netG_2_%d.pth' % (last_epoch), map_location=device)
    netG_1.load_state_dict(checkpoint_G_1)
    netG_2.load_state_dict(checkpoint_G_2)

    print("------------------------------------------------------")
    print("Successfully loaded SG-ShadowNet: ", last_epoch)
    print("------------------------------------------------------")

    rgb_dir_ws = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
    rgb_dir_ns = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
    rgb_dir_ws = rgb_dir_ws.format(dataset_version="v46_places")
    rgb_dir_ns = rgb_dir_ns.format(dataset_version="v46_places")
    save_dir_synth = "./reports/Synth/"

    ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
    mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"
    save_dir_istd = "./reports/ISTD/"

    ws_srd = "X:/SRD_Test/srd/shadow/*.jpg"
    ns_srd = "X:/SRD_Test/srd/shadow_free/*.jpg"
    mask_srd = "X:/SRD_Test/srd/mask/*.jpg"
    save_dir_srd = "./reports/SRD/"

    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, ws_istd, ns_istd, load_size, opts=opts)
    test_loader_istd = dataset_loader.load_istd_dataset(ws_istd, ns_istd, mask_istd, load_size, opts)
    test_loader_srd = dataset_loader.load_srd_dataset(ws_srd, ns_srd, mask_istd, load_size, opts)

    for i, (file_name, rgb_ws, rgb_ns, shadow_map, shadow_matte) in enumerate(train_loader, 0):
        netG_1.eval()
        netG_2.eval()

        s = rgb_ws.to(device)
        mask = shadow_matte.to(device)
        inv_mask = (1.0 - mask)

        fake_sf_temp = netG_1(s, mask)
        fake_sf = (s * inv_mask + fake_sf_temp * mask)
        fake_sf = netG_2(fake_sf, mask)
        results = fake_sf
        for j in range(0, np.size(file_name)):
            impath = save_dir_synth + file_name[j] + ".png"
            torchvision.utils.save_image(results[j], impath, normalize=True)
            print("Saving " +impath)

        break

    for i, (file_name, rgb_ws, rgb_ns, shadow_mask) in enumerate(test_loader_istd, 0):
        netG_1.eval()
        netG_2.eval()

        s = rgb_ws.to(device)
        mask = shadow_mask.to(device)
        inv_mask = (1.0 - mask)

        fake_sf_temp = netG_1(s, mask)
        fake_sf = (s * inv_mask + fake_sf_temp * mask)
        fake_sf = netG_2(fake_sf, mask)
        results = fake_sf

        for j in range(0, np.size(file_name)):
            impath = save_dir_istd + file_name[j] + ".png"
            torchvision.utils.save_image(results[j], impath, normalize=True)
            print("Saving " +impath)


    for i, (file_name, rgb_ws, rgb_ns, shadow_mask) in enumerate(test_loader_srd, 0):
        s = rgb_ws.to(device)
        mask = shadow_mask.to(device)
        inv_mask = (1.0 - mask)

        fake_sf_temp = netG_1(s, mask)
        fake_sf = (s * inv_mask + fake_sf_temp * mask)

        results = fake_sf

        resize_op = torchvision.transforms.Resize((160, 210), torchvision.transforms.InterpolationMode.BICUBIC)
        results = resize_op(results)

        for j in range(0, np.size(file_name)):
            impath = save_dir_srd + file_name[j] + ".png"
            torchvision.utils.save_image(results[j], impath, normalize=True)
            print("Saving " + impath)



if __name__ == "__main__":
    main()