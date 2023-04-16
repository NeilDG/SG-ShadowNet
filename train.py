from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
import torch

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
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training') #default 200
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

    last_epoch = 3
    checkpoint_G_1 = torch.load('ckpt_fs/netG_1_%d.pth' % (last_epoch), map_location=device)
    checkpoint_G_2 = torch.load('ckpt_fs/netG_2_%d.pth' % (last_epoch), map_location=device)
    netG_1.load_state_dict(checkpoint_G_1)
    netG_2.load_state_dict(checkpoint_G_2)
    opt.epoch = last_epoch

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

    opts = {}
    opts["img_to_load"] = 10000
    opts["num_workers"] = 6
    opts["cuda_device"] = "cuda:0"
    load_size = 20
    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, ws_istd, ns_istd, load_size, opts=opts)

    # Dataset loader
    # dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True),batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    # test_dataloader = DataLoader(TestImageDataset(opt.dataroot),batch_size= 1, shuffle=False, num_workers=opt.n_cpu)

    curr_iter = 0
    G_losses_temp = 0
    G_losses = []

    open(opt.log_path, 'w').write(str(opt) + '\n\n')

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        netG_1.train()
        netG_2.train()
        for i, (file_name, rgb_ws, rgb_ns, shadow_map, shadow_matte) in enumerate(train_loader, 0):
        # for i, (s, sgt,mask,mask50) in enumerate(dataloader):
            # Set model input
            # s = s.cuda()
            # sgt = sgt.cuda()
            # mask = mask.cuda()
            # mask50 = mask50.cuda()
            # inv_mask = (1.0- mask) # non_shadow region mask
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

        # Update learning rates
        lr_scheduler_G.step()

        # ##### testing ######
        # if epoch >= 0:
            # print('-------Start evaluation----------')
            # print('-------Total %d images ----------'%(len(test_dataloader)))
            # netG_1.eval()
            # netG_2.eval()
            # eval_shadow_rmse = 0
            # eval_nonshadow_rmse = 0
            # eval_rmse = 0
            # for j, (s, sgt, mask) in enumerate(test_dataloader):
                # s = s.cuda()                # full image shadow
                # sgt = sgt.cuda()            # full image shadow-free
                # mask = mask.cuda()          # shadow-mask
                # inv_mask = 1.0 - mask       # non_shadow region mask
                # with torch.no_grad():
                    # fake_sf_temp = netG_1(s, mask)
                    # input2 = (s * inv_mask + fake_sf_temp * mask)
                    # fake_sf = netG_2(input2, mask)
                    # # fake_sf = (s * inv_mask + fake_sf * mask)
                # mask = mask[0].cpu().float().numpy()[..., None][0, ...]
                # # evaluation
                # io.imsave('./ckpt_fs/sgt_val.png',tensor2img(sgt))
                # io.imsave('./ckpt_fs/fake_sf_val.png',tensor2img(fake_sf))
                # diff = calc_RMSE(tensor2img(sgt), tensor2img(fake_sf))
                # shadow_rmse = np.sqrt(1.0 * (np.power(diff, 2) * mask).sum(axis=(0, 1)) / mask.sum())
                # nonshadow_rmse = np.sqrt(1.0 * (np.power(diff, 2) * (1 - mask)).sum(axis=(0, 1)) / (1 - mask).sum())
                # whole_rmse = np.sqrt(np.power(diff, 2).mean(axis=(0, 1)))

                # eval_shadow_rmse += shadow_rmse.sum()
                # eval_nonshadow_rmse += nonshadow_rmse.sum()
                # eval_rmse += whole_rmse.sum()
            # eva_log = '[Eval] [Epoch] %d |rmse : %.3f | shadow_rmse : %.3f | nonshadow_rmse : %.3f' % \
                        # (epoch+1, eval_rmse / len(test_dataloader), eval_shadow_rmse / len(test_dataloader), eval_nonshadow_rmse / len(test_dataloader))
            # print(eva_log)
            # open(opt.log_path, 'a').write(eva_log + '\n')

        if epoch >= (opt.n_epochs-50):
            torch.save(netG_1.state_dict(), ('ckpt_fs/netG_1_%d.pth' % (epoch + 1)))
            torch.save(netG_2.state_dict(), ('ckpt_fs/netG_2_%d.pth' % (epoch + 1)))

        print('Epoch:{}'.format(epoch))

if __name__ == "__main__":
    main()