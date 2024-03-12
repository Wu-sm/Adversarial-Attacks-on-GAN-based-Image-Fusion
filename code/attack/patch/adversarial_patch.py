import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from attack.patch.adversarial_patch_util import *
from attack.attack_main2 import image_fusion

from skimage import transform



def train(epoch, patch, patch_shape, net, drawer, vgg,train_loader,device,save_dir,args, target_img):

    generator = net.decoder.to(device)
    generator.eval()
    encoder = net.encoder.to(device)
    encoder.eval()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data= Variable(data)

        # transform patch
        data_shape = data.data.cpu().numpy().shape
        if args.patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, args.image_size)
        elif args.patch_type == 'square':
            patch, mask = square_transform(patch, data_shape, patch_shape, args.image_size)
        patch, mask = torch.FloatTensor(patch).to(device), torch.FloatTensor(mask).to(device)
        patch, mask = Variable(patch), Variable(mask)

        adv_x, mask, patch, img_rec = attack(data, patch, mask,generator, encoder, vgg,device,args, target_img, save_dir,epoch, batch_idx)

        # if args.plot_all == 1 and batch_idx % 100 == 0:
        #     # plot source image
        #     # vutils.save_image(data.data, "%s/%d_%d_original.png" % (save_dir, epoch, batch_idx),
        #     #                   normalize=True)
        #     # # plot adversarial image
        #     vutils.save_image(adv_x.data, "%s/%d_%d_adversarial.png" % (save_dir, epoch, batch_idx),
        #                       normalize=True)
        #     vutils.save_image(img_rec.data, "%s/%d_%d_rec.png" % (save_dir, epoch, batch_idx),
        #                       normalize=True)
        #     torch.save(mask, os.path.join(save_dir, 'patch', '%d_%d_mask.npz' % (epoch, batch_idx)))
        #     torch.save(patch, os.path.join(save_dir, 'patch', '%d_%d_patch.npz' % (epoch, batch_idx)))

            # image_fusion(net, drawer, args.dataset_name, mask, patch, save_dir, '%d_%d_'% (epoch, batch_idx),device, args.is_cars)

        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch[i][j])

        patch = new_patch

        # log to file
        # progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.3f}".format(success / total))

    return patch, mask

# def use_other_patch(other_patch,other_patch_shape,data_shape, args, device):
#     patch = other_patch.data.cpu().numpy()
#     new_patch = np.zeros(other_patch_shape)
#     for i in range(new_patch.shape[0]):
#         for j in range(new_patch.shape[1]):
#             new_patch[i][j] = submatrix(patch[i][j])
#
#     if args.patch_type == 'circle':
#         patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, args.image_size)
#     elif args.patch_type == 'square':
#         patch, mask = square_transform(patch, data_shape, patch_shape, args.image_size)
#     patch, mask = torch.FloatTensor(patch).to(device), torch.FloatTensor(mask).to(device)
#     patch, mask = Variable(patch), Variable(mask)





def attack(img, patch, mask,generator, encoder, vgg, device,args, target_img, save_dir,epoch, batch_idx):
    output_file = os.path.join(save_dir, 'output_file.txt')
    output_file_loss = os.path.join(save_dir, 'output_file_loss.txt')
    output_file_loss_w = os.path.join(save_dir, 'w_loss.txt')
    resized_img = F.avg_pool2d(img, int(generator.size / 256), int(generator.size / 256))
    resized_img_target = F.avg_pool2d(target_img, int(generator.size / 256), int(generator.size / 256))
    with torch.no_grad():
        latent_org = encoder(resized_img)
        latent_target = encoder(resized_img_target)
        conv1_1_target, conv1_2_target, conv3_2_target, conv4_2_target = vgg(resized_img_target)
        conv1_1_org, conv1_2_org, conv3_2_org, conv4_2_org = vgg(resized_img)

    adv_x = torch.mul((1 - mask), img) + torch.mul(mask, patch)
    criterion = nn.MSELoss(reduction='mean').to(device)

    count = 0

    while True:
        count += 1
        adv_x = Variable(adv_x.data, requires_grad=True)

        adv_latent = encoder(F.avg_pool2d(adv_x.to(device), int(generator.size / 256), int(generator.size / 256)))

        adv_img_rec, _ = generator([encoder(F.avg_pool2d(adv_x,int(generator.size/256),int(generator.size/256)))], input_is_latent=True, randomize_noise=False,
                            return_latents=True)

        resized_adv_img_rec = F.avg_pool2d(adv_img_rec, int(generator.size / 256), int(generator.size / 256))
        conv1_1_rec, conv1_2_rec, conv3_2_rec, conv4_2_rec = vgg(resized_adv_img_rec)
        l_lpips_rec_adv_target = criterion(conv1_1_rec, conv1_1_target) + criterion(conv1_2_rec,conv1_2_target) + criterion(conv3_2_rec,conv3_2_target) + criterion(conv4_2_rec, conv4_2_target)
        l_latent_target_adv = criterion(latent_target, adv_latent)
        l_latent_org_adv = criterion(latent_org, adv_latent)
        l_img_adv_rec_target = criterion(target_img, adv_img_rec)
        Loss = 0*l_latent_target_adv - 1*l_latent_org_adv + 0*l_img_adv_rec_target + 0*l_lpips_rec_adv_target
        # Loss =  l_latent_target_adv - 10*l_latent_org_adv
        # Loss = 1 * l_lpips_rec_adv_target + 2 * l_img_adv_rec_target
        Loss.backward()

        adv_grad = adv_x.grad.clone()

        adv_x.grad.data.zero_()

        patch -= adv_grad

        adv_x = torch.mul((1 - mask), img) + torch.mul(mask, patch)
        adv_x = torch.clamp(adv_x, torch.min(img), torch.max(img))


        with open(output_file_loss_w, 'a') as f:
            f.write('Loss:%.5f\n' % Loss)
        if args.save_img and (count==1 or count==args.max_count): #(count%10==0 or count==args.max_count)
            # plot adversarial image
            # vutils.save_image(adv_x.data, "%s/%d_%d_%d_adversarial.png" % (save_dir,epoch, batch_idx,count),
            #                   normalize=True)
            # vutils.save_image(adv_img_rec.data, "%s/%d_%d_%d_rec.png" % (save_dir,epoch, batch_idx,count),
            #                   normalize=True)
            print('epoch:%d batch_idx:%d count:%d l_latent_target:%.5f;   l_latent_org:%.5f;     l_img_rec_target:%f   l_lpips_rec_adv_target:%f' % (epoch,batch_idx,
            count, l_latent_target_adv, l_latent_org_adv, l_img_adv_rec_target, l_lpips_rec_adv_target))
            with open(output_file, 'a') as f:
                f.write('%dth img count: %d l_latent_target:%.5f;   l_latent_org:%.5f;     l_img_rec_target:%f   l_lpips_rec_adv_target:%f\n' % (
                batch_idx, count, l_latent_target_adv, l_latent_org_adv, l_img_adv_rec_target, l_lpips_rec_adv_target))
        with open(output_file_loss, 'a') as f:
            f.write(
                '%dth img count: %d loss:%.5f\n' % (batch_idx, count, Loss))
        if count >= args.max_count:
            break

    return adv_x, mask, patch,adv_img_rec


def main(drawer, net, vgg,train_dataloader, device,save_dir,args, target_img):

    # mask = torch.load(
    #     os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/data', 'aa_ffhq_0.100_mask.npz')).to(
    #     device)
    # patch = torch.load(
    #     os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/data', 'aa_ffhq_0.100_patch.npz')).to(
    #     device)
    # FFHQ
    # 正常
    # mask = torch.load(os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/ffhq/58_ffhq_patch_white_box_2000_50_0.100 (full ok)/adversarial/patch','ffhq_2000_0.100_mask.npz')).to(
    #     device)
    # patch = torch.load(os.path.join(
    #     '/home/sh/pycharm_workspace/StyleFusion-main/runs/ffhq/58_ffhq_patch_white_box_2000_50_0.100 (full ok)/adversarial/patch',
    #     'ffhq_2000_0.100_patch.npz')).to(
    #     device)
    # 初始化的，未经优化
    # mask = torch.load(os.path.join(
    #     '/home/sh/pycharm_workspace/StyleFusion-main/runs/ffhq/58_ffhq_patch_white_box_2000_50_0.100 (full ok)/adversarial/patch/1_0_mask.npz')).to(
    #     device)
    # patch = torch.load(os.path.join(
    #     '/home/sh/pycharm_workspace/StyleFusion-main/runs/ffhq/58_ffhq_patch_white_box_2000_50_0.100 (full ok)/adversarial/patch','1_0_patch.npz')).to(
    #     device)

    #use ffhq patch for the car
    # mask = transform.rescale(mask.cpu(), [1,1,0.5,0.5])
    # patch = transform.rescale(patch.cpu() , [1,1,0.5,0.5])

    # # #car
    # mask = torch.load(os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/car/13_car_patch_white_box_2000_50_0.100/adversarial/patch','1_1100_mask.npz')).to(
    #     device)
    # patch = torch.load(os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/car/13_car_patch_white_box_2000_50_0.100/adversarial/patch','1_1100_patch.npz')).to(
    #     device)
    # 优化的
    # mask = torch.load(os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/car/13_car_patch_white_box_2000_50_0.100/adversarial/patch','1_0_mask.npz')).to(
    #     device)
    # patch = torch.load(os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/car/13_car_patch_white_box_2000_50_0.100/adversarial/patch','1_0_patch.npz')).to(
    #     device)
    # random patch
    # if args.patch_type == 'circle':
    #     patch, patch_shape = init_patch_circle(args.image_size, args.patch_size)
    # elif args.patch_type == 'square':
    #     patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
    # data_shape = [1, 3, args.image_size, args.image_size]
    # patch, mask = square_transform(patch, data_shape, patch_shape, args.image_size)
    # patch, mask = torch.FloatTensor(patch).to(device), torch.FloatTensor(mask).to(device)

    # # 消融
    mask = torch.load(os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/ffhq/214_ffhq_patch_white_box_2000_50_0.100/adversarial/patch', 'ffhq_2000_0.100_mask.npz')).to(device)
    patch = torch.load(os.path.join('/home/sh/pycharm_workspace/StyleFusion-main/runs/ffhq/214_ffhq_patch_white_box_2000_50_0.100/adversarial/patch', 'ffhq_2000_0.100_patch.npz')).to(device)
    return torch.tensor(patch).to(device), torch.tensor(mask).to(device)


    if args.patch_type == 'circle':
        patch, patch_shape = init_patch_circle(args.image_size, args.patch_size)
    elif args.patch_type == 'square':
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
    else:
        sys.exit("Please choose a square or circle patch")

    for epoch in range(1, args.epochs + 1):
        # patch = train(epoch, patch, patch_shape)
        patch, mask = train(epoch, patch, patch_shape, net, drawer, vgg, train_dataloader, device, save_dir,args, target_img)


    #change numpy to tensor
    # transform path
    # patch_shape = patch.shape
    data_shape = [1,3,args.image_size,args.image_size]
    if args.patch_type == 'circle':
        patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, args.image_size)
    elif args.patch_type == 'square':
        patch, mask = square_transform(patch, data_shape, patch_shape, args.image_size)
    patch, mask = torch.FloatTensor(patch).to(device), torch.FloatTensor(mask).to(device)

    torch.save(mask, os.path.join(save_dir, 'patch', '%s_%d_%.3f_mask.npz' % (args.dataset_name, args.train_size, args.patch_size)))
    torch.save(patch, os.path.join(save_dir, 'patch', '%s_%d_%.3f_patch.npz' % (args.dataset_name, args.train_size,args.patch_size)))

    # torch.save(mask, os.path.join(args.save_dir, 'data', '%s_%d_%.3f_mask.npz' % (args.dataset_name, args.train_size, args.patch_size)))
    # torch.save(patch, os.path.join(args.save_dir, 'data', '%s_%d_%.3f_patch.npz' % (args.dataset_name, args.train_size,args.patch_size)))
    return patch, mask