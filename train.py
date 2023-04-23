from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader
import torch
from yaml import SafeLoader

from adapter import dataset_loader
from utils.utils import LambdaLR
from utils.utils import weights_init_normal, tensor2img, calc_RMSE
from models.model import ConGenerator_S2F, ConRefineNet
from loss.losses import L_spa
from data.datasets import ImageDataset, TestImageDataset
import numpy as np
from skimage import io,color
from skimage.transform import resize
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training') #default 200
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='epoch to start linearly decaying the learning rate to 0') #default 50
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
opt = parser.parse_args()


# ISTD datasets
opt.dataroot = 'input/dataset/ISTD'

# checkpoint dir
if not os.path.exists('ckpt_fs'):
    os.mkdir('ckpt_fs')
# opt.log_path = os.path.join('ckpt_fs', str(datetime.datetime.now()) + '.txt')
opt.log_path = os.path.join('ckpt_fs', 'train_logs.txt')

print(opt)

def main():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    ###### Definition of variables ######
    # Networks
    netG_1 = ConGenerator_S2F()
    netG_2 = ConRefineNet()

    netG_1.cuda()
    netG_2.cuda()

    netG_1.apply(weights_init_normal)
    netG_2.apply(weights_init_normal)

    # last_epoch = 0
    # checkpoint_G_1 = torch.load('ckpt_fs/netG_1_%d.pth' % (last_epoch), map_location=device)
    # checkpoint_G_2 = torch.load('ckpt_fs/netG_2_%d.pth' % (last_epoch), map_location=device)
    # netG_1.load_state_dict(checkpoint_G_1)
    # netG_2.load_state_dict(checkpoint_G_2)
    # opt.epoch = last_epoch

    print("------------------------------------------------------")
    print("Successfully loaded SG-ShadowNet: ", opt.epoch)
    print("------------------------------------------------------")

    # Lossess
    # criterion_GAN = torch.nn.MSELoss()  # lsgan
    # criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
    criterion_region = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_spa = L_spa()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_1.parameters(), netG_2.parameters()),lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    ### data loader
    rgb_dir_ws = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
    rgb_dir_ns = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
    rgb_dir_ws = rgb_dir_ws.format(dataset_version="v46_places")
    rgb_dir_ns = rgb_dir_ns.format(dataset_version="v46_places")

    ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
    mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"

    ws_srd = "X:/SRD_Test/srd/shadow/*.jpg"
    ns_srd = "X:/SRD_Test/srd/shadow_free/*.jpg"
    mask_srd = "X:/SRD_Test/srd/mask_srd/*.jpg"

    opts = {}
    opts["img_to_load"] = 10000
    opts["num_workers"] = 12
    opts["cuda_device"] = "cuda:0"
    load_size = 10
    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, ws_istd, ns_istd, load_size, opts=opts)
    test_loader_istd = dataset_loader.load_istd_dataset(ws_istd, ns_istd, mask_istd, load_size, opts)
    # test_loader_istd = dataset_loader.load_srd_dataset(ws_srd, ns_srd, mask_srd, load_size, opts)
    # test_loader_srd = dataset_loader.load_srd_dataset(ws_srd, ns_srd, mask_srd, load_size, opts)

    # Dataset loader
    # dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True),batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    # test_dataloader = DataLoader(TestImageDataset(opt.dataroot),batch_size= 1, shuffle=False, num_workers=opt.n_cpu)

    curr_iter = 0
    G_losses_temp = 0
    G_losses = []

    open(opt.log_path, 'w').write(str(opt) + '\n\n')

    # plot utils
    plot_loss_path = "./reports/train_test_loss.yaml"
    l1_loss = nn.L1Loss()
    if (os.path.exists(plot_loss_path)):
        with open(plot_loss_path) as f:
            losses_dict = yaml.load(f, SafeLoader)
    else:
        losses_dict = {}
        losses_dict["train"] = []
        losses_dict["test_istd"] = []

    print("Losses dict: ", losses_dict["train"])
    current_step = 0

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        netG_1.train()
        netG_2.train()
        for i, (file_name, rgb_ws, rgb_ns, shadow_map, shadow_matte) in enumerate(train_loader, 0):
            current_step += 1
            s = rgb_ws.to(device)
            sgt = rgb_ns.to(device)
            mask = shadow_matte.to(device)
            mask50 = mask
            inv_mask = (1.0 - mask)

            ###### Generators ######
            optimizer_G.zero_grad()

            fake_sf_temp = netG_1(s, mask)
            loss_1 = criterion_identity(fake_sf_temp, sgt)
            loss_shadow1 = criterion_region(torch.cat(((fake_sf_temp[:,0]+1.0)*mask50-1.0,fake_sf_temp[:,1:]*mask50),1),torch.cat(((sgt[:,0]+1.0)*mask50-1.0,sgt[:,1:]*mask50),1))
            input2 = (s * inv_mask + fake_sf_temp * mask)

            output = netG_2(input2,mask)
            loss_2 = criterion_identity(output,sgt)
            loss_shadow2 = criterion_region(torch.cat(((output[:,0]+1.0)*mask50-1.0,output[:,1:]*mask50),1),torch.cat(((sgt[:,0]+1.0)*mask50-1.0,sgt[:,1:]*mask50),1))
            loss_spa = torch.mean(criterion_spa(output, sgt)) *10
            # Total loss
            loss_G = loss_1 + loss_2 + loss_shadow1 + loss_shadow2 + loss_spa
            loss_G.backward()

            G_losses_temp += loss_G.item()

            optimizer_G.step()
            ###################################

            curr_iter += 1

            if (i+1) % opt.iter_loss == 0:
                log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_1 %.5f], [loss_2 %.5f], [loss_shadow1 %.5f], [loss_shadow2 %.5f]' % \
                      (epoch, curr_iter, loss_G,loss_1,loss_2,loss_spa,loss_shadow2)
                print(log)
                open(opt.log_path, 'a').write(log + '\n')

                G_losses.append(G_losses_temp / opt.iter_loss)
                G_losses_temp = 0

                avg_log = '[the last %d iters], [loss_G %.5f]'% (opt.iter_loss, G_losses[G_losses.__len__()-1])
                print(avg_log)
                open(opt.log_path, 'a').write(avg_log + '\n')

                slabimage=output.data
                save_dir="./ckpt_fs/"
                impath = save_dir + file_name[0] + ".png"
                torchvision.utils.save_image(slabimage[0], impath, normalize=True)

                torch.save(netG_1.state_dict(), ('ckpt_fs/netG_1_%d.pth' % (epoch + 1)))
                torch.save(netG_2.state_dict(), ('ckpt_fs/netG_2_%d.pth' % (epoch + 1)))

            if(current_step % 500 == 0):
                netG_1.eval()
                netG_2.eval()

                # plot train-test loss
                fake_sf_temp = netG_1(s, mask)
                input2 = (s * inv_mask + fake_sf_temp * mask)
                rgb_ns_like = netG_2(input2, mask)

                train_loss = float(np.round(l1_loss(rgb_ns_like, sgt).item(), 4))
                losses_dict["train"].append({current_step: float(train_loss)})

                #test istd
                _, rgb_ws, rgb_ns, shadow_matte = next(itertools.cycle(test_loader_istd))
                s = rgb_ws.to(device)
                sgt = rgb_ns.to(device)
                mask = shadow_matte.to(device)
                inv_mask = (1.0 - mask)

                fake_sf_temp = netG_1(s, mask)
                input2 = (s * inv_mask + fake_sf_temp * mask)
                rgb_ns_like = netG_2(input2, mask)
                # rgb_ns_like = fake_sf_temp

                test_loss_istd = float(np.round(l1_loss(rgb_ns_like, sgt).item(), 4))
                losses_dict["test_istd"].append({current_step: float(test_loss_istd)})

                plot_loss_file = open(plot_loss_path, "w")
                yaml.dump(losses_dict, plot_loss_file)
                plot_loss_file.close()
                print("Dumped train test loss to ", plot_loss_path)

        # Update learning rates
        lr_scheduler_G.step()

        if epoch >= (opt.n_epochs-50):
            torch.save(netG_1.state_dict(), ('ckpt_fs/netG_1_%d.pth' % (epoch + 1)))
            torch.save(netG_2.state_dict(), ('ckpt_fs/netG_2_%d.pth' % (epoch + 1)))

        print('Epoch:{}'.format(epoch))

if __name__ == "__main__":
    main()