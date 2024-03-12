from matplotlib import pyplot as plt
from style_fusion_simple import StyleFusionSimple, tensor2im

import argparse
import torch
import numpy as np
import sys
import os
import dlib
import cv2
import torch.nn.functional as F
import attack.patch.adversarial_patch as patch
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from dataset.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face
from PIL import Image
import torchvision.utils as vutils
from torchvision import transforms
import lpips
import torch.optim as optim
import torch.nn as nn
import glob
import random
import legacy
import dnnlib
from skimage import io, color, metrics


from vgg.vgg import vgg16

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def stylefusion():
    device = 'cuda:0'
    ffhq_drawer = StyleFusionSimple('ffhq',
                                    '/home/sh/pycharm_workspace/encoder4editing-main/pretrained_models/stylegan2-ffhq-config-f.pt',
                                    'weights/ffhq_weights.json', device)

    z_mounth = ffhq_drawer.seed_to_z((6, 7))
    z_background = ffhq_drawer.seed_to_z((23, 8))
    z_hair = ffhq_drawer.seed_to_z((334, 6))
    z_eyes = ffhq_drawer.seed_to_z((337, 5))
    z_global = ffhq_drawer.seed_to_z((393, 5))

    with torch.no_grad():
        I_background, inner_feature = ffhq_drawer.generate_img(z_background)
        I_mounth, inner_feature = ffhq_drawer.generate_img(z_mounth)
        I_hair, inner_feature = ffhq_drawer.generate_img(z_hair)
        I_eyes, inner_feature = ffhq_drawer.generate_img(z_eyes)
        I_global, inner_feature = ffhq_drawer.generate_img(z_global)
        I_fused, inner_feature = ffhq_drawer.generate_img(z_global, hair=z_hair, eyes=z_eyes, background=z_background,
                                                          mouth=z_mounth)

    plt.figure(figsize=(5 * 5, 5))
    plt.axis('off')
    plt.imshow(tensor2im(torch.cat([I_background, I_hair, I_eyes, I_mounth, I_global], dim=2)))
    plt.axis('off')
    plt.imshow(tensor2im(I_fused))


def inversion(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    # if os.path.exists(latents_file_path):
    #     latent_codes = torch.load(latents_file_path).to(device)
    # else:
    #     latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
    #     torch.save(latent_codes, latents_file_path)

    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)
    torch.save(latent_codes, latents_file_path)

    if not args.latents_only:
        generate_inversions(args, generator, latent_codes, is_cars=is_cars)


def setup_data_loader(args, opts, train_batch_size, test_batch_size):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    # idx
    idx = np.arange(len(test_dataset))
    np.random.shuffle(idx)
    train_idx = idx[:args.train_size]
    test_idx = idx[args.train_size:args.test_size + args.train_size]

    train_data_loader = DataLoader(test_dataset,
                                   batch_size=train_batch_size,
                                   shuffle=False,
                                   num_workers=2,
                                   drop_last=True,
                                   sampler=SubsetRandomSampler(train_idx))

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  drop_last=True,
                                  sampler=SubsetRandomSampler(test_idx))

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, train_data_loader, test_data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


# def get_all_latents(net, data_loader, n_images=None, is_cars=False):
#     all_latents = []
#     i = 0
#     with torch.no_grad():
#         for batch in data_loader:
#             if n_images is not None and i > n_images:
#                 break
#             x = batch
#             inputs = x.to(device).float()
#             latents = get_latents(net, inputs, is_cars)
#             all_latents.append(latents)
#             i += len(latents)
#     return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    if isinstance(idx, int):
        im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    else:
        im_save_path = os.path.join(save_dir, "%s.jpg" % idx)
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    print('Saving inversion images')
    inversions_directory_path = os.path.join(args.save_dir, 'inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(min(args.n_sample, len(latent_codes))):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)


def run_alignment(image_path):
    predictor = dlib.shape_predictor(paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'cars_' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    ffhq_drawer = StyleFusionSimple('ffhq',
                                    '/home/sh/pycharm_workspace/encoder4editing-main/pretrained_models/stylegan2-ffhq-config-f.pt',
                                    'weights/ffhq_weights.json', device=device)

    # dataset
    all_latents = []
    all_imgs = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if args.n_sample is not None and i > args.n_sample:
                break
            x = batch
            inputs = x.to(device).float()

            if i == 5:
                noise = add_noise(inputs, magnitude=555)
                # eps=30
                # noise = rotate(inputs, eps, resize=False)
                latents = get_latents(net, noise, is_cars)
            else:
                latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            all_imgs.append(inputs)
            i += len(latents)
    all_latents = torch.cat(all_latents)
    all_imgs = torch.cat(all_imgs)

    # save_image(inputs, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 'org')
    # save_image(noise, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 'noisy')

    vutils.save_image((all_imgs + 1) / 2, '/home/sh/pycharm_workspace/StyleFusion-main/runs/org.jpg')

    z_mounth, z_background, z_hair, z_eyes, z_global = all_latents[1:6]

    # z_mounth = ffhq_drawer.seed_to_z((6,7))
    # z_background = ffhq_drawer.seed_to_z((23,8))
    # z_hair = ffhq_drawer.seed_to_z((334,6))
    # z_eyes = ffhq_drawer.seed_to_z((337,5))
    # z_global = ffhq_drawer.seed_to_z((393,5))

    z_mounth = z_mounth.unsqueeze(dim=0)
    z_background = z_background.unsqueeze(dim=0)
    z_hair = z_hair.unsqueeze(dim=0)
    z_eyes = z_eyes.unsqueeze(dim=0)
    z_global = z_global.unsqueeze(dim=0)

    with torch.no_grad():
        I_fused = ffhq_drawer.generate_img(z_global, hair=z_hair, eyes=z_eyes, background=z_background, mouth=z_mounth,
                                           latents_type="w")
        I_mounth = ffhq_drawer.generate_img(z_mounth, latents_type="w")
        I_background = ffhq_drawer.generate_img(z_background, latents_type="w")
        I_hair = ffhq_drawer.generate_img(z_hair, latents_type="w")
        I_eyes = ffhq_drawer.generate_img(z_eyes, latents_type="w")
        I_global = ffhq_drawer.generate_img(z_global, latents_type="w")

    I_all = torch.cat([I_mounth, I_background, I_hair, I_eyes, I_global, I_fused], dim=2)
    save_image(I_all, args.save_dir, 'noise_rotate111')
    vutils.save_image((I_all + 1) / 2, os.path.join(args.save_dir, 'I_all.jpg'))

    # plt.figure(figsize=(5*5,5))
    # plt.axis('off')
    # plt.imshow(tensor2im(torch.cat([I_background,I_hair,I_eyes, I_mounth, I_global,I_fused],dim=2)))
    # plt.axis('off')
    # plt.imshow(tensor2im(I_fused))

    # save_image(I_background, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 0)
    # save_image(I_hair, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 1)
    # save_image(I_eyes, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 2)
    # save_image(I_mounth, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 3)
    # save_image(I_global, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 4)
    # save_image(I_fused, '/home/sh/pycharm_workspace/StyleFusion-main/runs', 5)


def add_noise(imgs, magnitude):
    imgs_blur = imgs.numpy().transpose(0, 2, 3, 1)
    imgs_noisy = []
    for i in range(imgs.shape[0]):
        blur = cv2.GaussianBlur(imgs_blur[i], (magnitude, magnitude), 0)
        imgs_noisy.append(blur)
    imgs_noisy = np.array(imgs_noisy)
    imgs_noisy = imgs_noisy.transpose(0, 3, 1, 2)
    imgs_noisy = torch.from_numpy(imgs_noisy)
    return imgs_noisy

def dp_noise(inputs,args, device):
    # Sensitivity of the query or computation (a positive number)
    # sensitivity = 1.0
    # # Privacy parameter (a small positive number, typically denoted as "epsilon")
    # epsilon = 1.0
    # # Calculate the scale (b) for Laplace noise
    # scale = sensitivity / epsilon
    # The data to which you want to add noise
    # Generate Laplace noise and add it to the data
    noisy_data = inputs + torch.tensor(np.random.laplace(0, args.scale, size=inputs.numel())).to(device).reshape(inputs.size())

    # Now 'noisy_data' contains the original data with Laplace noise added
    # print(f"Noisy Data: {noisy_data}")
    return noisy_data.float()

def main_optimize(inputs,drawer,net, target_img, args, device, iter_dict, train_dataloader, save_dir):
    adversarial = args.adversarial
    generator = net.decoder.to(device)
    generator.eval()
    encoder = net.encoder.to(device)
    encoder.eval()

    all_adv_inputs = []

    if adversarial == 'dp_noise':
        adv_inputs = dp_noise(inputs,args, device)
        all_adv_inputs.append(adv_inputs)

    if adversarial == 'patch':
        paste_size = generator.size // args.paste_times
        location = (generator.size - paste_size) // 2
        with open(param_file, 'a') as f:
            f.write("paste_size {}\n".format(paste_size))
            f.write("location {}\n".format(location))
        adv_inputs = adversarial_patch(inputs, target_img, net, drawer, paste_size, location, device, save_dir, is_cars)
        all_adv_inputs.append(adv_inputs)

    # patch that the adversary has knowledge of generator and encoder so that optimize the patch
    if adversarial == 'patch_white_box':
        args.image_size = inputs.size(3)
        new_run_folder(os.path.join(save_dir,'patch'))
        #to attain optimized mask and patch; or load the existing ones
        adv_patch, mask = patch.main(drawer,net, vgg, train_dataloader, device, save_dir, args, target_img)
        adv_patch, mask = adv_patch.to(device), mask.to(device)
        adv_inputs = patch_white_box(inputs, mask, adv_patch)#, save_dir, net, drawer, is_cars)
        all_adv_inputs.append(adv_inputs)

    if adversarial == 'white_box_target' or adversarial == 'white_box_patch':
        with open(param_file, 'a') as f:
            f.write("white_box")
            f.write("n_iters {}\n".format(iter_dict[generator.size]))
        print('white-box')
        if adversarial == 'white_box_target':
            target_img = target_img  # img_transform(target_img).unsqueeze(0).cuda()
        else:
            paste_size = generator.size // args.paste_times
            location = (generator.size - paste_size) // 2
            with open(param_file, 'a') as f:
                f.write("paste_size {}\n".format(paste_size))
                f.write("location {}\n".format(location))
            all_target_img = []
            for i in range(inputs.size(0)):
                all_target_img.append(
                    get_paste_image(inputs[i].clone().unsqueeze(dim=0), target_img, location, paste_size, save_dir,
                                    'paste_target_img_%d' % i))
            all_target_img = torch.cat((all_target_img), dim=0)
            vutils.save_image((all_target_img + 1) / 2, os.path.join(save_dir, 'target_img.jpg'))
            target_img = all_target_img
        adv_inputs = white_box(inputs, target_img, drawer, net, vgg, args, n_iters=iter_dict[generator.size],
                               is_cars=is_cars, save_dir=save_dir)
        all_adv_inputs.append(adv_inputs)
        # adv_inputs = []
        # #add perturbation
        # for i in range(inputs.size(0)):
        #     if i>-1:
        #         # adv_img = optimize(net, inputs[i].clone().unsqueeze(dim=0), img2, args.save_dir, device, n_iters=300)
        #         # target_img = get_paste_image(inputs[i].clone().unsqueeze(dim=0), img2, int(generator.size/2), args.save_dir, 'paste_target_img_%d'%i)
        #         adv_img = optimize_vgg(net, vgg, inputs[i].clone().unsqueeze(dim=0), target_img, args.save_dir, device, n_iters=iter_dict[generator.size])
        #         adv_inputs.append(adv_img)
        #     else:
        #         adv_inputs.append(inputs[i].unsqueeze(dim=0))
        # adv_inputs = torch.cat((adv_inputs), dim=0)
        #
        # noise = adv_inputs - inputs
        # vutils.save_image((adv_inputs + 1) / 2, os.path.join(args.save_dir, 'adv_inputs.jpg'))
        # vutils.save_image((noise + 1) / 2, os.path.join(args.save_dir, 'noise.jpg'))
        #
        # #fusion after noise
        # all_adv_latents = get_latents(net, F.avg_pool2d(adv_inputs,int(generator.size/256),int(generator.size/256)), is_cars)
        # fusion(dataset_name=args.dataset_name, all_latents = all_adv_latents, drawer = drawer, save_dir = args.save_dir, file_name = 'adv_fusion')

    # gradually adding an out-domain image
    if adversarial == 'out_domain_more':
        # target_img = img_transform(target_img)#.unsqueeze(0).cuda()
        for i in range(inputs.size(0)):
            inputs[i, :, :, :] = target_img
        adv_inputs = inputs
        all_adv_inputs.append(adv_inputs)
        # vutils.save_image((inputs + 1) / 2, os.path.join(save_dir, 'adv_inputs_%d.jpg'%i))
        # # fusion after noise
        # all_latents = get_latents(net, F.avg_pool2d(inputs, int(inputs.size(2) / 256), int(inputs.size(2) / 256)),
        #                               is_cars)
        # fusion(dataset_name=args.dataset_name, all_latents=all_latents, drawer=drawer, save_dir=save_dir,
        #        file_name='adv_fusion_%d'%i)

    # once an out-domain image
    if adversarial == 'out_domain_single':
        # target_img = img_transform(target_img)#.unsqueeze(0).cuda()
        for i in range(inputs.size(0)):
            adv_inputs = inputs.clone()
            adv_inputs[i, :, :, :] = target_img
            all_adv_inputs.append(adv_inputs)
            # vutils.save_image((adv_inputs + 1) / 2, os.path.join(save_dir, 'adv_inputs_%d.jpg'%i))
            # # fusion after noise
            # all_latents = get_latents(net, F.avg_pool2d(adv_inputs, int(adv_inputs.size(2) / 256), int(adv_inputs.size(2) / 256)),
            #                               is_cars)
            # fusion(dataset_name=args.dataset_name, all_latents=all_latents, drawer=drawer, save_dir=save_dir,
            #        file_name='adv_fusion_%d'%i)
            # calculate_distance(net, all_latents)
            # print('aaaaaa')
    return all_adv_inputs


def new_run_folder(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    return file_dir


def patch_white_box(inputs, mask, adv_patch):#, save_dir, net, drawer, is_cars):
    all_adv_inputs = []
    for i in range(inputs.size(0)):
        adv_inputs = torch.mul((1 - mask), inputs[i]) + torch.mul(mask, adv_patch)
        # adv_x = torch.clamp(adv_x, 0, 1)
        adv_inputs = torch.clamp(adv_inputs, torch.min(inputs[i]), torch.max(inputs[i]))
        all_adv_inputs.append(adv_inputs)
    all_adv_inputs = torch.cat(all_adv_inputs, dim=0)

    # vutils.save_image((all_adv_inputs.detach() + 1) / 2, os.path.join(save_dir, 'all_adv_inputs.png'))
    # all_adv_latents = get_latents(net, F.avg_pool2d(all_adv_inputs, int(net.decoder.size / 256),
    #                                                 int(net.decoder.size / 256)), is_cars)
    # fusion(dataset_name=args.dataset_name, all_latents=all_adv_latents, drawer=drawer, save_dir=save_dir,
    #        file_name='adv_fusion')

    # vutils.save_image((inputs.detach() + 1) / 2, os.path.join(save_dir, 'inputs.png'))
    # all_latents = get_latents(net, F.avg_pool2d(inputs, int(net.decoder.size / 256), int(net.decoder.size / 256)),
    #                           is_cars)
    # fusion(dataset_name=args.dataset_name, all_latents=all_latents, drawer=drawer, save_dir=save_dir,
    #        file_name='fusion')
    return all_adv_inputs


def adversarial_patch(inputs, paste_image, net, drawer, paste_size, location, device, save_dir, is_cars):
    generator_size = net.decoder.size
    paste_inputs = []
    for i in range(inputs.size(0)):
        paste_img = get_paste_image(inputs[i].clone().unsqueeze(dim=0), paste_image, location=location,
                                    paste_size=paste_size, save_dir=save_dir,
                                    filename='paste_target_img_%d' % i)
        paste_inputs.append(paste_img)
    paste_inputs = torch.cat((paste_inputs), dim=0)

    return paste_inputs


def get_paste_image(background_tensor, image_to_paste_tensor, location, paste_size, save_dir, filename):
    img_transform = transforms.Compose([
        transforms.Resize((paste_size, paste_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image_to_paste_tensor = img_transform(transforms.ToPILImage()(image_to_paste_tensor.squeeze(dim=0))).unsqueeze(
        0).to(device)
    # Define the position where you want to paste the image (top-left corner)
    paste_position = (location, location)
    # Paste the image onto the background
    background_tensor[:, :, paste_position[0]:paste_position[0] + image_to_paste_tensor.shape[2],
    paste_position[1]:paste_position[1] + image_to_paste_tensor.shape[3]] = image_to_paste_tensor
    # vutils.save_image((background_tensor + 1) / 2, os.path.join(save_dir, '%s.jpg'%filename))
    return background_tensor


def white_box(inputs, target_img, drawer, net, vgg, args, n_iters, is_cars, save_dir):
    img_size = inputs.size(2)
    adv_inputs = []
    # add perturbation
    if len(args.which_adv)==0:
        args.which_adv = [x for x in range(inputs.size(0))]

    for i in range(inputs.size(0)):
        if i in args.which_adv:  # only the first one to be adversarial
            if target_img.size(0) == 1:  # white-box target
                adv_img = optimize_vgg(i, net, vgg, inputs[i].clone().unsqueeze(dim=0), target_img, save_dir, device,
                                       args=args, file_name='optimize_%d' % i,
                                       n_iters=n_iters)
            else:  # white-box patch
                adv_img = optimize_vgg(i, net, vgg, inputs[i].clone().unsqueeze(dim=0), target_img[i].unsqueeze(dim=0),
                                       save_dir, device, args=args,
                                       file_name='optimize_%d' % i,
                                       n_iters=n_iters)
            adv_inputs.append(adv_img)
        else:
            adv_inputs.append(inputs[i].unsqueeze(dim=0))
    adv_inputs = torch.cat((adv_inputs), dim=0)

    # noise = adv_inputs - inputs
    # vutils.save_image((adv_inputs + 1) / 2, os.path.join(save_dir, 'adv_inputs.jpg'))
    # vutils.save_image((noise + 1) / 2, os.path.join(save_dir, 'noise.jpg'))
    #
    # # fusion after noise
    # all_adv_latents = get_latents(net, F.avg_pool2d(adv_inputs, int(img_size / 256), int(img_size / 256)),
    #                               is_cars)
    # fusion(dataset_name=args.dataset_name, all_latents=all_adv_latents, drawer=drawer, save_dir=save_dir,
    #        file_name='adv_fusion')

    return adv_inputs


def calculate_distance(net, latent):
    MSE_criterion = nn.MSELoss(reduction='none').to(device)
    distance = torch.mean(MSE_criterion(net.latent_avg, latent), dim=(1, 2))

    return distance


# for those datasets we have no real images
def generate_images(drawer, n_imgs):
    all_I = []
    for i in range(n_imgs):
        z = drawer.seed_to_z((np.random.randint(1, 1000, 2)))
        with torch.no_grad():
            I, _ = drawer.generate_img(z, latents_type="z")
        all_I.append(I)
        # vutils.save_image((I + 1) / 2, os.path.join(args.save_dir, 'car_fusion.jpg'))
    all_I = torch.cat((all_I), dim=0)
    return all_I


def fusion(dataset_name, all_latents, drawer, save_dir='/', file_name='filename', feature_idx=-1):
    all_inner_feature = []
    # extracted_values = [inner_feature[idx] for idx in feature_idx]

    if 'ffhq' in dataset_name:
        z_mounth, z_background, z_hair, z_eyes, z_global = all_latents.unsqueeze(dim=1)
        with torch.no_grad():
            I_fused, inner_feature = drawer.generate_img(z_global, hair=z_hair, eyes=z_eyes, background=z_background,
                                                         mouth=z_mounth,
                                                         latents_type="w")

        with torch.no_grad():
            I_mounth, inner_feature = drawer.generate_img(z_mounth, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_background, inner_feature = drawer.generate_img(z_background, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_hair, inner_feature = drawer.generate_img(z_hair, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_eyes, inner_feature = drawer.generate_img(z_eyes, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_global, inner_feature = drawer.generate_img(z_global, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
        I_all = torch.cat([I_mounth, I_background, I_hair, I_eyes, I_global], dim=0)
        # vutils.save_image((I_all+1)/2,os.path.join(save_dir, 'without_%s.jpg'%file_name))

    if 'car' in dataset_name:
        z_wheel, z_background_top, z_background_bottom,z_body = all_latents.unsqueeze(dim=1)
        with torch.no_grad():
            I_fused, inner_feature = drawer.generate_img(z_body, wheels=z_wheel, bg_top=z_background_top,
                                                         bg_bottom=z_background_bottom,
                                                         latents_type="w")

        with torch.no_grad():
            I_mounth, inner_feature = drawer.generate_img(z_body, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_background, inner_feature = drawer.generate_img(z_wheel, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_hair, inner_feature = drawer.generate_img(z_background_top, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_eyes, inner_feature = drawer.generate_img(z_background_bottom, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
        I_all = torch.cat([I_mounth, I_background, I_hair, I_eyes], dim=0)
        # vutils.save_image((I_all+1)/2,os.path.join(save_dir, 'without_%s.jpg'%file_name))

    if 'church' in dataset_name:
        z_background_top, z_background_bottom,z_body = all_latents.unsqueeze(dim=1)
        with torch.no_grad():
            I_fused, inner_feature = drawer.generate_img(z_body, bg_top=z_background_top, bg_bottom=z_background_bottom,
                                                         latents_type="w")
        with torch.no_grad():
            I_mounth, inner_feature = drawer.generate_img(z_body, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_hair, inner_feature = drawer.generate_img(z_background_top, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
            I_eyes, inner_feature = drawer.generate_img(z_background_bottom, latents_type="w")
            all_inner_feature.append(inner_feature[feature_idx])
        I_all = torch.cat([I_mounth, I_hair, I_eyes], dim=0)
        # vutils.save_image((I_all + 1) / 2, os.path.join(save_dir, 'without_%s.jpg' % file_name))

    # vutils.save_image((I_fused + 1) / 2, os.path.join(save_dir, '%s.jpg' % file_name))
    return I_fused, I_all, torch.cat(all_inner_feature, dim=0)


def optimize_vgg(i_th_img, Model, vgg, img, img_target, run_dir, device, file_name, args,
                 n_iters=1000):  # images:tensor,training images

    decoder, encoder = Model.decoder, Model.encoder
    img_org = img.clone().detach()
    # img_target = (img_target+img_org)/2
    resized_img_org = F.avg_pool2d(img_org, int(decoder.size / 256), int(decoder.size / 256))
    resized_img_target = F.avg_pool2d(img_target, int(decoder.size / 256), int(decoder.size / 256))
    img.requires_grad = True

    output_file = os.path.join(run_dir, 'optimize_output.txt')

    with torch.no_grad():
        latent_target = encoder(resized_img_target)
        latent_org = encoder(resized_img_org)

        # target_rec, _ = decoder([get_latents(Model, F.avg_pool2d(img_target,int(decoder.size/256),int(decoder.size/256)))], input_is_latent=True, randomize_noise=False,
        #                     return_latents=True)
        conv1_1_target, conv1_2_target, conv3_2_target, conv4_2_target = vgg(resized_img_target)
        conv1_1_org, conv1_2_org, conv3_2_org, conv4_2_org = vgg(resized_img_org)

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam([{'params': img}], lr=args.lr)  # 0.01

    # lpips_loss = lpips.LPIPS(net='vgg').to(device)

    # img_all = img_org.cpu()
    # rec_img_all= img_org.cpu()

    # print("The Latent Code is being optimized...")
    for iters in range(n_iters):
        optimizer.zero_grad()
        # pred = decoder(encoder(img))
        # img_rec, _ = decoder([get_latents(Model, F.avg_pool2d(img,int(decoder.size/256),int(decoder.size/256)))], input_is_latent=True, randomize_noise=False,
        #                     return_latents=True)
        img_rec, _ = decoder([encoder(F.avg_pool2d(img, int(decoder.size / 256), int(decoder.size / 256)))],
                             input_is_latent=True, randomize_noise=False,
                             return_latents=True)
        latent_pred = encoder(F.avg_pool2d(img, int(decoder.size / 256), int(decoder.size / 256)))
        resized_img_rec = F.avg_pool2d(img_rec, int(decoder.size / 256), int(decoder.size / 256))
        resized_img = F.avg_pool2d(img, int(decoder.size / 256), int(decoder.size / 256))
        conv1_1_rec, conv1_2_rec, conv3_2_rec, conv4_2_rec = vgg(resized_img_rec)
        l_lpips_rec_target = criterion(conv1_1_rec, conv1_1_target) + criterion(conv1_2_rec,
                                                                                conv1_2_target) + criterion(conv3_2_rec,
                                                                                                            conv3_2_target) + criterion(
            conv4_2_rec, conv4_2_target)
        l_lpips_rec_org = criterion(conv1_1_rec, conv1_1_org) + criterion(conv1_2_rec, conv1_2_org) + criterion(
            conv3_2_rec, conv3_2_org) + criterion(conv4_2_rec, conv4_2_org)
        conv1_1_img, conv1_2_img, conv3_2_img, conv4_2_img = vgg(resized_img)
        l_lpips_img = criterion(conv1_1_img, conv1_1_org) + criterion(conv1_2_img, conv1_2_org) + criterion(conv3_2_img,
                                                                                                            conv3_2_org) + criterion(
            conv4_2_img, conv4_2_org)
        # l_lpips_img =  criterion(conv4_2_img, conv4_2_org)
        # l_lpips_rec_org = lpips_loss(resized_img_target, resized_img_rec)
        l_img_rec_target = criterion(img_target, img_rec)
        # l_img_rec = criterion(img_org, img_rec)
        # img is close to the original image
        l_img_org = criterion(img_org, img)
        # l_img_target = criterion(resized_img_target, resized_img)
        # l_lpips_org = lpips_loss(resized_img_org, resized_img)
        l_latent_target = criterion(latent_target, latent_pred)
        l_latent_org = criterion(latent_org, latent_pred)
        # inversion_loss = 10*l_img_org  + l_latent + l_img_rec -criterion(latent_org, latent_pred)+l_lpips_rec #+l_lpips_img
        # inversion_loss =  10*l_latent_target + 10*l_lpips_rec_org -l_latent_org + 5*l_img_org +0.1*l_lpips_rec_org #+l_lpips_img
        # omit the rec
        inversion_loss = 10 * l_latent_target + l_img_rec_target - l_latent_org + 20 * l_img_org + l_lpips_img #+ 0.1 * l_lpips_rec_target  # +l_lpips_img
        # inversion_loss = l_latent + l_img_rec + l_lpips_rec
        # loss = criterion(img, pred)
        inversion_loss.backward(retain_graph=True)
        optimizer.step()
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # print("\r[iter: %d/%d] [pixel-wise loss: %.2f]" % (iters, n_iters, inversion_loss.item()), end='')

        if args.save_img and (iters % 5) == 0 and (iters // 5) > 0:
            # img_all = torch.cat([img_all, img.cpu().clone()])
            # rec_img_all =  torch.cat([rec_img_all, img_rec.cpu().clone()])
            vutils.save_image((img_rec.detach() + 1) / 2, os.path.join(run_dir, 'rec_%s_%d.png' % (file_name, iters)))
            vutils.save_image((img + 1) / 2, os.path.join(run_dir, 'adv_input_%s_%d.png' % (file_name, iters)))
            print('iter: %d l_latent_target:%.5f;   l_latent_org:%.5f;     l_img_org:%f' % (
            iters, l_latent_target, l_latent_org, l_img_org))
            with open(output_file, 'a') as f:
                f.write('%dth img iter: %d l_latent_target:%.5f;   l_latent_org:%.5f;     l_img_org:%f \n' % (
                i_th_img, iters, l_latent_target, l_latent_org, l_img_org))

    # save inverted images
    # vutils.save_image((img_all.detach() + 1) / 2, os.path.join(run_dir,'optimize.png'))
    # vutils.save_image((rec_img_all.detach() + 1) / 2, os.path.join(run_dir, 'optimize_rec.png'))
    return img.detach()


def optimize(Model, img, img_target, run_dir, device, n_iters=1000):  # images:tensor,training images

    decoder, encoder = Model.decoder, Model.encoder
    img_org = torch.tensor(img)
    resized_img_org = F.avg_pool2d(img_org, 4, 4)
    resized_img_target = F.avg_pool2d(img_target, 4, 4)
    img.requires_grad = True
    latent_target = encoder(F.avg_pool2d(img_target.to(device), 4, 4))
    target_rec, _ = decoder([encoder(F.avg_pool2d(img_target, 4, 4))], input_is_latent=True, randomize_noise=False, )

    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = optim.Adam([{'params': img}], lr=0.01)

    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    img_all = img_org

    # print("The Latent Code is being optimized...")
    for iters in range(n_iters):
        optimizer.zero_grad()
        # pred = decoder(encoder(img))
        img_rec, _ = decoder([encoder(F.avg_pool2d(img, 4, 4))], input_is_latent=True, randomize_noise=False,
                             return_latents=True)
        latent_pred = encoder(F.avg_pool2d(img, 4, 4))
        resized_img_rec = F.avg_pool2d(img_rec, 4, 4)
        resized_img = F.avg_pool2d(img, 4, 4)
        # l_lpips_rec = lpips_loss(img_target, img_rec)
        l_lpips_rec = lpips_loss(resized_img_target, resized_img_rec)
        l_img_rec = criterion(img_target, img_rec)
        # img is close to the original image
        l_img_org = criterion(img_org, img)
        l_lpips_org = lpips_loss(resized_img_org, resized_img)
        l_latent = criterion(latent_target, latent_pred)
        inversion_loss = l_img_org + l_lpips_rec + l_latent + l_img_rec  # +l_lpips_org
        # loss = criterion(img, pred)
        inversion_loss.backward(retain_graph=True)
        optimizer.step()
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # print("\r[iter: %d/%d] [pixel-wise loss: %.2f]" % (iters, n_iters, inversion_loss.item()), end='')

        if (iters % 50) == 0 and (iters // 50) > 0:
            img_all = torch.cat([img_all, img.clone()])

    # save inverted images
    vutils.save_image((img_all.detach() + 1) / 2, os.path.join(run_dir, 'optimize.png'))
    return img.clone().detach()


def optimize_copy(Model, img, img_target, run_dir, device, n_iters=1000):  # images:tensor,training images

    decoder, encoder = Model.decoder, Model.encoder
    img_org = torch.tensor(img)
    img.requires_grad = True
    latent_target = encoder(F.avg_pool2d(img_target.to(device), 4, 4))
    target_rec, _ = decoder([encoder(F.avg_pool2d(img_target, 4, 4))], input_is_latent=True, randomize_noise=False, )

    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = optim.Adam([{'params': img}], lr=0.01)

    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    img_all = img_org

    # print("The Latent Code is being optimized...")
    for iters in range(n_iters):
        optimizer.zero_grad()
        # pred = decoder(encoder(img))
        img_rec, _ = decoder([encoder(F.avg_pool2d(img, 4, 4))], input_is_latent=True, randomize_noise=False,
                             return_latents=True)
        latent_pred = encoder(F.avg_pool2d(img, 4, 4))
        # resized_pred = F.avg_pool2d(pred, 4, 4)
        # l_lpips = lpips_loss(img_org, pred)
        l_lpips = lpips_loss(img_target, img_rec)
        l_img_rec = criterion(target_rec, img_rec)
        l_img = criterion(img_org, img)
        l_latent = criterion(latent_target, latent_pred)
        inversion_loss = l_img + l_lpips + l_latent + l_img_rec
        # loss = criterion(img, pred)
        inversion_loss.backward(retain_graph=True)
        optimizer.step()
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # print("\r[iter: %d/%d] [pixel-wise loss: %.2f]" % (iters, n_iters, inversion_loss.item()), end='')

        if (iters % 50) == 0 and (iters // 50) > 0:
            img_all = torch.cat([img_all, img.clone()])

    # save inverted images
    vutils.save_image((img_all.detach() + 1) / 2, os.path.join(run_dir, 'optimize.png'))
    return img.detach()


def cal_rec_loss(img, rec_img):
    # criterion = nn.MSELoss(reduction='mean').to(device)  # sum mean
    # rec_loss = criterion(img, rec_img)

    squared_diff = (img - rec_img) ** 2
    mse = squared_diff.mean(dim=[1, 2, 3])

    return mse


# def get_models(args):
#
#
#     net, opts = setup_model(args.ckpt%args.dataset_name, device)
#     args, train_dataloader, test_dataloader = setup_data_loader(args, opts,batch_size=1)
#     return net, train_dataloader, test_dataloader

def new_adv_dir(base_dir,postfix):
    filenames = sorted(glob.glob(base_dir+ os.path.sep + '*' + os.path.sep), key=os.path.getctime)

    if len(filenames) > 0:
        num = int(filenames[-1].split('/')[-2].split('_')[0]) + 1
    else:
        num = 0
    final_dir = os.path.join(base_dir, '%d_%s' % (num, postfix))
    while os.path.exists(final_dir):
        num = num + 1
        final_dir = os.path.join(base_dir, '%d_%s' % (num, postfix))
    return new_run_folder(final_dir)

def image_fusion(net, drawer,dataset_name,mask,adv_patch,save_dir,file_fix,device,is_cars):
    # inputs = torch.load(os.path.join(
    #     '/home/sh/pycharm_workspace/StyleFusion-main/runs/data',
    #     'car_input.npz')).float().to(device)
    inputs = torch.load(os.path.join(
        '/home/sh/pycharm_workspace/StyleFusion-main/runs/data',
        'car_input.npz')).float().to(device)
    adv_inputs = patch_white_box(inputs, mask, adv_patch)

    all_latents = get_latents(net, F.avg_pool2d(adv_inputs, int(net.decoder.size / 256),
                                                int(net.decoder.size / 256)),
                              is_cars)
    I_fused, rec_inputs, inner_feature = fusion(dataset_name=dataset_name, all_latents=all_latents,
                                                drawer=drawer)

    vutils.save_image((adv_inputs + 1) / 2, os.path.join(save_dir, 'adv_inputs_%s.jpg' % file_fix))
    vutils.save_image((I_fused + 1) / 2, os.path.join(save_dir, 'fusion_%s.jpg' % file_fix))
    vutils.save_image((rec_inputs + 1) / 2, os.path.join(save_dir, 'fusion_without_%s.jpg' % file_fix))

def get_training_data_dir(dataset_name):
    if 'car' in dataset_name.lower():
        # return '/home/sh/pycharm_workspace/StyleFusion-main/runs/data/generated_data/car'
        return '/home/sh/pycharm_workspace/dataset/stanfordCars/cars_test/cars_test'
    elif 'ffhq' in dataset_name.lower():
        return '/home/sh/pycharm_workspace/dataset/ffhq'
    elif 'church' in dataset_name.lower():
        return 'sorry'

def cal_SSMI(original_image, distorted_image):

    original_image = (original_image.permute(1, 2, 0)).numpy()
    distorted_image = (distorted_image.permute(1, 2, 0)).numpy()

    if original_image.shape != distorted_image.shape:
        raise ValueError("Both images must have the same dimensions and shape.")

    # Convert the images to grayscale if they are in color
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        original_image = color.rgb2gray(original_image)
    if len(distorted_image.shape) == 3 and distorted_image.shape[2] == 3:
        distorted_image = color.rgb2gray(distorted_image)

    # Calculate the SSIM between the two images
    ssim_score = metrics.structural_similarity(original_image, distorted_image)
    return ssim_score


if __name__ == "__main__":
    device = "cuda:0"

    setup_seed(123456789)
    # stylefusion()

    parser = argparse.ArgumentParser(description="Inference")
    # parser.add_argument("--images_dir", type=str, default='/home/sh/pycharm_workspace/dataset/ffhq/00000',
    #                     help="The directory of the images to be inverted")
    parser.add_argument("--images_dir", type=str, default='/home/sh/pycharm_workspace/dataset/ffhq',
                        help="The directory of the images to be inverted")
    parser.add_argument("--dataset_name", type=str, default='ffhq',  # car, ffhq, church
                        help="dataset name")
    parser.add_argument("--save_dir", type=str, default='/home/sh/pycharm_workspace/StyleFusion-main/runs',
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=5, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=6, help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true", help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true", help="align face images before inference")
    # parser.add_argument("ckpt", metavar="CHECKPOINT", default='/home/sh/pycharm_workspace/encoder4editing-main/datasets/e4e_ffhq_encode.pt', help="path to generator checkpoint")
    parser.add_argument("--ckpt", type=str,
                        default='/home/sh/pycharm_workspace/encoder4editing-main/datasets/e4e_%s_encode.pt',
                        help="path to generator checkpoint")

    # patch
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--max_count', type=int, default=50,
                        help='max number of iterations to find adversarial example')
    parser.add_argument('--patch_type', type=str, default='square', help='patch type: circle or square')
    parser.add_argument('--patch_size', type=float, default=0.1, help='patch size. E.g. 0.05 ~= 5% of image ')  # 0.05
    parser.add_argument('--train_size', type=int, default=2000, help='Number of training images')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test images')
    parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images')
    parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')
    parser.add_argument('--regenerate', type=int, default=1, help='regenerate a new patch:1')

    # white-box target
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of the optimization')  #
    parser.add_argument('--which_adv', type=list, default=[0, 1, 2, 3, 4], help='which image is to be perturbed')  #
    # dataset_n_dict = {'ffhq': 5, 'car': 4, 'church': 3}  0, 1, 2, 3, 4

    parser.add_argument('--use_generate_img', type=bool, default=False, help='use generated images for fusion task')
    parser.add_argument('--paste_times', type=int, default=3, help='the size of the paste 1/paste_times')
    parser.add_argument('--save_img', type=bool, default=True, help='use generated images for fusion task')

    #for hybrid attack
    parser.add_argument('--hybrid_adv', type=bool, default=False, help='use hybrid adversarial examples')
    #hybrid_adv_from_existing means not to start new adversarial examples
    parser.add_argument('--hybrid_adv_from_existing', type=bool, default=False, help='use hybrid adversarial examples from existing dirs')
    parser.add_argument('--hybrid_adv_dirs',nargs='*', type=str, default=['1_ffhq_patch_white_box_0.200','2_ffhq_white_box_target_100_0.00500_[0,1,2,3,4]' ], help='use hybrid adversarial examples')

    #dp noise
    parser.add_argument('--scale', type=float, default=0.4, help='scale of dp noise,0-0.1')  # 0.05


    args = parser.parse_args()




    # all_dataset=['ffhq', 'car', 'church']
    all_dataset = ['car']
    adversarial_choose = ['patch_white_box']
    # adversarial_choose = ['patch_white_box','out_domain_more', 'white_box_target', 'white_box_patch','patch']

    # iter_dict = {1024: 1000, 512: 1000, 256: 500}
    iter_dict = {1024: 100, 512: 100, 256: 50}
    dataset_n_dict = {'ffhq': 5, 'car': 4, 'church': 3}

    # load vgg model
    vgg = vgg16(os.path.join("/home/sh/pycharm_workspace/stylegan2-ada-unlearn/vgg", "imagenet_vgg16.pth")).to(device)
    vgg.eval()

    # target image
    target_img = Image.open('/home/sh/pycharm_workspace/StyleFusion-main/dataset/target_imgs/vase1.png')
    # target_img = Image.open('/home/sh/pycharm_workspace/StyleFusion-main/dataset/target_imgs/vase2.png')

    # attack
    for dataset in all_dataset:
        # new a folder fo the dataset
        dataset_savedir = new_run_folder(os.path.join(args.save_dir, dataset))
        args.dataset_name = dataset
        args.images_dir = get_training_data_dir(dataset)
        net, opts = setup_model(args.ckpt % args.dataset_name, device)
        is_cars = 'cars_' in opts.dataset_type
        args.is_cars = is_cars
        args, train_dataloader, test_dataloader = setup_data_loader(args, opts, train_batch_size=1,
                                                                    test_batch_size=dataset_n_dict[args.dataset_name])
        drawer = StyleFusionSimple(args.dataset_name, None,
                                   '/home/sh/pycharm_workspace/StyleFusion-main/weights/%s_weights.json' % args.dataset_name,
                                   device=device, GAN=net.decoder)

        network_pkl = '/home/sh/pycharm_workspace/stylegan2-ada-pytorch-main/pre-trained/ffhq.pkl'
        with dnnlib.util.open_url(network_pkl) as f:
            models = legacy.load_network_pkl(f)
            # G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
            D = models['D'].to(device)  # type: ignore

        # for target img
        img_transform = transforms.Compose([
            transforms.Resize((net.decoder.size, net.decoder.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        target_img = img_transform(target_img).unsqueeze(0).cuda()


        #如果从已有的攻击中来做混合，那么强制清空攻击门类，不再发起新的攻击
        if args.hybrid_adv and args.hybrid_adv_from_existing:
            adversarial_choose=[]
        #collect each adversarial dir
        all_attack_dir = []
        for adversarial in adversarial_choose:

            args.adversarial = adversarial

            #make a dir for the adversarial attack
            if adversarial == 'patch':
                adversarial_postfix =  '%s_%s_%d' % (dataset, adversarial, args.paste_times)
            elif adversarial == 'patch_white_box':
                adversarial_postfix = '%s_%s_%d_%d_%.3f' % (dataset, adversarial,args.train_size, args.max_count, args.patch_size)
            elif adversarial == 'white_box_target' or adversarial == 'white_box_patch':
                which_adv = ",".join([str(item) for item in args.which_adv])
                adversarial_postfix = '%s_%s_%d_%.5f_[%s]' % (
                    dataset, adversarial, iter_dict[net.decoder.size], args.lr, which_adv)
            else:
                adversarial_postfix = '%s_%s' % (dataset, adversarial)
            attack_savedir = new_adv_dir(dataset_savedir, adversarial_postfix)
            all_attack_dir.append(adversarial_postfix)

            # new folder for benign and adversarial inputs and fusion
            benign_savedir = new_run_folder(os.path.join(attack_savedir, 'benign'))
            adv_savedir = new_run_folder(os.path.join(attack_savedir, 'adversarial'))

            # parameter file
            param_file = os.path.join(attack_savedir, 'parameters.txt')
            with open(param_file, 'a') as f:
                f.write("adversarial attack {}\n".format(adversarial))
                f.write("dataset {}\n".format(args.dataset_name))
                f.write("dataset size {}\n".format(net.decoder.size))
                f.write("epochs {}\n".format(args.epochs))
                f.write("max_count {}\n".format(args.max_count))
                f.write("patch_size {}\n".format(args.patch_size))
                f.write("train_size {}\n".format(args.train_size))
                f.write("patch_type {}\n".format(args.patch_type))
                f.write("white-box max_iter {}\n".format(iter_dict[net.decoder.size]))
                f.write("white-box lr {}\n".format(args.lr))
                f.write("use_generate_img {}\n".format(args.use_generate_img))

            all_adv_inputs = []
            all_inputs = []
            all_rec_loss = []
            all_adv_rec_loss = []
            all_inner_feature = []
            all_adv_inner_feature = []

            # images for fusion
            for batch_idx, batch in enumerate(test_dataloader):  # dataset_n_dict[args.dataset_name] as a batch
                if batch_idx > 0:
                    break
                if args.use_generate_img:
                    inputs = generate_images(drawer, n_imgs=dataset_n_dict[args.dataset_name])
                else:
                    inputs = batch.float().to(device)
                #FFHQ
                # inputs = torch.load(os.path.join(
                #     '/home/sh/pycharm_workspace/StyleFusion-main/runs/data',
                #     'all_inputs.npz')).float().to(device)
                # vutils.save_image((inputs + 1) / 2, os.path.join(benign_savedir, 'org_inputs_%d.jpg'%batch_idx))
                # original fusion
                all_latents = get_latents(net, F.avg_pool2d(inputs, int(net.decoder.size / 256),
                                                            int(net.decoder.size / 256)),
                                          is_cars)
                I_fused, rec_inputs, inner_feature = fusion(dataset_name=args.dataset_name, all_latents=all_latents,
                                                            drawer=drawer, save_dir=benign_savedir,
                                                            file_name='org_fusion_%d' % batch_idx, feature_idx=-1)

                all_inputs.append(inputs.cpu())
                # inner feature
                all_inner_feature.append(inner_feature.cpu())
                # rec loss of benign samples
                rec_loss = cal_rec_loss(inputs, rec_inputs)
                all_rec_loss.append(rec_loss.cpu())
                if args.save_img:
                    vutils.save_image((inputs + 1) / 2, os.path.join(benign_savedir, 'org_inputs_%d.jpg' % batch_idx))
                    vutils.save_image((I_fused + 1) / 2, os.path.join(benign_savedir, 'org_fusion_%d.jpg' % batch_idx))
                    vutils.save_image((rec_inputs + 1) / 2,
                                      os.path.join(benign_savedir, 'org_without_fusion_%d.jpg' % batch_idx))
                    # with torch.no_grad():
                    #     print('inputs:%s'% str(D(inputs,c=None).tolist()))
                    #     print('I_fused:%s' % str(D(I_fused, c=None).tolist()))
                    #     print('rec_inputs:%s' % str(D(rec_inputs, c=None).tolist()))

                # attain adversarial examples
                # adv_inputs = main_optimize(net, target_img, args, device, iter_dict, test_dataloader, adv_savedir)
                adv_inputs = main_optimize(inputs.clone(), drawer, net, target_img, args, device, iter_dict, train_dataloader, adv_savedir)
                for i in range(len(adv_inputs)):
                    all_adv_inputs.append(adv_inputs[i].cpu())
                    all_adv_latents = get_latents(net,
                                                  F.avg_pool2d(adv_inputs[i], int(net.decoder.size / 256),
                                                               int(net.decoder.size / 256)),
                                                  is_cars)
                    # adv_img_rec, _ = net.decoder(
                    #     [net.encoder(F.avg_pool2d(adv_inputs[i], int(net.decoder.size / 256), int(net.decoder.size / 256)))],
                    #     input_is_latent=True, randomize_noise=False,
                    #     return_latents=True)
                    # vutils.save_image((adv_img_rec + 1) / 2,
                    #                   os.path.join(adv_savedir, 'adv_rec_%d.jpg'%i))


                    #partially craft an adversarial example one at a time
                    all_partial_fused_img = []
                    for j in range(all_adv_latents.size(0)+1):
                        if j<all_adv_latents.size(0): #once an adversarial
                            partial_adv_inputs = inputs.clone()
                            partial_adv_inputs[j] = adv_inputs[i][j]
                            partial_adv_latent_codes = all_latents.clone()
                            partial_adv_latent_codes[j] = all_adv_latents[j]
                            file_postfix = '%d_%d_%d' % (batch_idx, i, j)
                        else:# all are adversarial
                            partial_adv_inputs = adv_inputs[i].clone()
                            partial_adv_latent_codes = all_adv_latents.clone()
                            file_postfix = '%d_%d_all' % (batch_idx, i)

                        adv_I_fused, adv_rec_inputs, adv_inner_feature = fusion(dataset_name=args.dataset_name,
                                                                            all_latents=partial_adv_latent_codes, drawer=drawer,
                                                                            save_dir=adv_savedir,
                                                                            file_name='adv_fusion_'+file_postfix)

                        all_partial_fused_img.append(adv_I_fused.cpu())

                        if j == all_adv_latents.size(0): #只有全部都是adversarial examples的时候才算
                            # rec loss of benign samples
                            adv_rec_loss = cal_rec_loss(adv_inputs[i], adv_rec_inputs)
                            all_adv_rec_loss.append(adv_rec_loss.cpu())
                            # inner feature
                            all_adv_inner_feature.append(adv_inner_feature.cpu())
                        if args.save_img:
                            vutils.save_image((partial_adv_inputs + 1) / 2,
                                              os.path.join(adv_savedir, 'adv_inputs_'+file_postfix+'.jpg'))
                            vutils.save_image((adv_I_fused + 1) / 2,
                                              os.path.join(adv_savedir, 'adv_fusion_'+file_postfix+'.jpg'))
                            vutils.save_image((adv_rec_inputs + 1) / 2,
                                              os.path.join(adv_savedir, 'adv_without_fusion_'+file_postfix+'.jpg'))
                            if j == all_adv_latents.size(0):
                                vutils.save_image((torch.cat(all_partial_fused_img,dim=0) + 1) / 2,
                                                  os.path.join(adv_savedir, 'partial_fusion_'+file_postfix+'.jpg'))

                                print(cal_SSMI(I_fused.squeeze(0).cpu(), adv_I_fused.squeeze(0).cpu()))

                                # with torch.no_grad():
                                #     print('adv_inputs:%s' % str(D(partial_adv_inputs, c=None).tolist()))
                                #     print('adv_I_fused:%s' % str(D(I_fused, c=None).tolist()))
                                #     print('adv_rec_inputs:%s' % str(D(rec_inputs, c=None).tolist()))

                if batch_idx % 5 == 0:
                    torch.save(torch.cat(all_adv_inputs, dim=0), os.path.join(adv_savedir, 'all_adv_inputs.npz'))
                    torch.save(torch.cat(all_inputs, dim=0), os.path.join(benign_savedir, 'all_inputs.npz'))
                    torch.save(torch.cat(all_adv_rec_loss, dim=0), os.path.join(adv_savedir, 'all_adv_rec_loss.npz'))
                    torch.save(torch.cat(all_rec_loss, dim=0), os.path.join(benign_savedir, 'all_rec_loss.npz'))
                    torch.save(torch.cat(all_adv_inner_feature, dim=0),
                               os.path.join(adv_savedir, 'all_adv_inner_feature.npz'))
                    torch.save(torch.cat(all_inner_feature, dim=0),
                               os.path.join(benign_savedir, 'all_inner_feature.npz'))

            torch.save(torch.cat(all_adv_inputs, dim=0), os.path.join(adv_savedir, 'all_adv_inputs.npz'))
            torch.save(torch.cat(all_inputs, dim=0), os.path.join(benign_savedir, 'all_inputs.npz'))
            torch.save(torch.cat(all_adv_rec_loss, dim=0), os.path.join(adv_savedir, 'all_adv_rec_loss.npz'))
            torch.save(torch.cat(all_rec_loss, dim=0), os.path.join(benign_savedir, 'all_rec_loss.npz'))
            torch.save(torch.cat(all_adv_inner_feature, dim=0), os.path.join(adv_savedir, 'all_adv_inner_feature.npz'))
            torch.save(torch.cat(all_inner_feature, dim=0), os.path.join(benign_savedir, 'all_inner_feature.npz'))


        ###############################################################################
        #hybrid attack   all_attack_dir   dataset_n_dict[args.dataset_name]
        if args.hybrid_adv==True:
            hybrid_attack_savedir = new_adv_dir(dataset_savedir,'%s_hybrid_attack' % dataset)
            # hybrid_attack_savedir = new_run_folder(
            #     os.path.join(dataset_savedir, '%s_hybrid_attack' % dataset))
            hybrid_param_file = os.path.join(hybrid_attack_savedir, 'hybrid_param.txt')
            with open(hybrid_param_file, 'a') as f:
                f.write("dataset {}\n".format(args.dataset_name))
                f.write("dataset size {}\n".format(net.decoder.size))
            if args.hybrid_adv_from_existing:
                all_attack_dir = args.hybrid_adv_dirs
            if len(all_attack_dir) == 0:
                    break
            examples_per_dir = dataset_n_dict[args.dataset_name] // len(all_attack_dir)
            remainder = dataset_n_dict[args.dataset_name] % len(all_attack_dir)
            # Create a list to store the number of apples each child gets
            examples_for_dir = [examples_per_dir] * len(all_attack_dir)
            # Distribute the remaining apples to the first 'remainder' children
            for i in range(remainder):
                examples_for_dir[i] += 1
            hybrid_examples = []
            start_index=0
            for i in range(len(all_attack_dir)):
                all_adv_inputs = torch.load(os.path.join(dataset_savedir, all_attack_dir[i],'adversarial','all_adv_inputs.npz'))
                hybrid_examples.append(all_adv_inputs[start_index:start_index+examples_for_dir[i]])
                start_index = start_index + examples_for_dir[i]
                with open(hybrid_param_file, 'a') as f:
                    f.write("attacks {}: {}\n".format(i,all_attack_dir[i]))
            all_latents = get_latents(net, F.avg_pool2d(torch.cat(hybrid_examples,dim=0).to(device), int(net.decoder.size / 256),
                                                        int(net.decoder.size / 256)),is_cars)
            I_fused, rec_inputs, inner_feature = fusion(dataset_name=args.dataset_name, all_latents=all_latents,
                                                        drawer=drawer, save_dir=hybrid_attack_savedir,
                                                        file_name='org_fusion', feature_idx=-1)
            vutils.save_image((torch.cat(hybrid_examples, dim=0) + 1) / 2,
                              os.path.join(hybrid_attack_savedir, 'hybrid_fusion_inputs.jpg'))
            vutils.save_image((I_fused + 1) / 2,
                              os.path.join(hybrid_attack_savedir, 'hybrid_fusion.jpg'))
