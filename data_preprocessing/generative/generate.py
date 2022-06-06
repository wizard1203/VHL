import logging
import argparse
from copy import deepcopy
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import utils
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm


import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt



sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


from utils.distribution_utils import train_distribution_diversity
from loss_fn.cov_loss import (
    cov_non_diag_norm, cov_norm
)

from utils.matrix_utils import orthogo_tensor

from model.build import create_model



import time

def generate(args, device):

    g_ema = create_model(args, args.model, output_dim=10).to(device)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    noise_num = args.noise_num

    n_dim = g_ema.num_layers
    normed_n_mean = train_distribution_diversity(
        n_distribution=noise_num, n_dim=n_dim, max_iters=500)
    style_GAN_latent_noise_mean = normed_n_mean.detach().to(device)
    style_GAN_latent_noise_std = [0.1 / n_dim]*n_dim

    global_zeros = torch.ones((noise_num, args.style_gan_style_dim)) * 0.0
    global_mean_vector = torch.normal(mean=global_zeros, std=args.style_gan_sample_z_mean)
    style_GAN_sample_z_mean = global_mean_vector
    style_GAN_sample_z_std = args.style_gan_sample_z_std

    g_ema.eval()

    # TODO, for more efficiently loading data.
    # if args.package:

    with torch.no_grad():
        for noise_i in tqdm(range(args.noise_num)):
            iters = args.sample // args.batch_size

            train = "train"
            class_dir = f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}"

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            for batch_i in tqdm(range(iters)):
                mean_vector = style_GAN_sample_z_mean[noise_i].repeat(args.batch_size, 1)
                sample_z = torch.normal(mean=mean_vector, std=style_GAN_sample_z_std).to(device)

                latent_noise_mean_i = style_GAN_latent_noise_mean[noise_i]

                sample, _ = g_ema(
                    [sample_z], truncation=args.truncation, truncation_latent=mean_latent,
                    noise_mean=latent_noise_mean_i, noise_std=style_GAN_latent_noise_std
                )
                print(f"sample.shape: {sample.shape}")
                for j in range(args.batch_size):
                    index = batch_i * args.batch_size + j 
                    utils.save_image(
                        sample[j],
                        f"{class_dir}/{str(index).zfill(6)}.jpg"
                    )
            utils.save_image(
                sample,
                f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}-overview.jpg",
                nrow=args.overview_n_columns
            )


def train_generator_diversity(device, generator, max_iters=100, min_loss=0.0):
    generator.train()
    generator.to(device)
    for i in range(max_iters):
        generator_optimizer = torch.optim.SGD(generator.parameters(),
            lr=0.01, weight_decay=0.0001, momentum=0.9)
        means = torch.zeros((64, args.vae_decoder_z_dim))
        z = torch.normal(mean=means, std=1.0).to(device)
        data = generator(z)
        loss_diverse = cov_non_diag_norm(data)
        generator_optimizer.zero_grad()
        loss_diverse.backward()
        generator_optimizer.step()
        print(f"Iteration: {i}, loss_diverse: {loss_diverse.item()}")
        if loss_diverse.item() < min_loss:
            print(f"Iteration: {i}, loss_diverse: {loss_diverse.item()} smaller than min_loss: {min_loss}, break")
            break
    generator.cpu()



def generate_from_vae_decoder(args, device):

    generator_dict = {}
    for i in range(args.noise_num):
        generator = create_model(args,
            model_name=args.model, output_dim=args.noise_num)
        generator_dict[i] = generator

    for i in range(args.noise_num):
        generator = generator_dict[i]
        train_generator_diversity(device, generator, 50, -1.0)

    with torch.no_grad():
        for noise_i in tqdm(range(args.noise_num)):
            iters = args.sample // args.batch_size
            train = "train"
            class_dir = f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}"
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            generator = generator_dict[noise_i]
            generator.eval()
            generator.to(device)
            for batch_i in tqdm(range(iters)):
                means = torch.zeros((args.batch_size, args.vae_decoder_z_dim))
                noise = torch.normal(mean=means, std=1.0).to(device)
                sample = generator(noise)
                # torch.mean(a)
                # torch.std(a)
                # data = (data - data.mean()) / data.max()
                for _ in range(3):
                    data_channel = sample[:,_,:,:]
                    # data_channel = (data_channel - data_channel.mean()) / data_channel.max()
                    # logging.debug(f"data std: {torch.std(data[:,_,:,:])} mean: {torch.mean(data[:,_,:,:])}")
                    logging.info(f"data std: {torch.std(data_channel)} mean: {torch.mean(data_channel)}")
                for j in range(args.batch_size):
                    index = batch_i * args.batch_size + j 
                    utils.save_image(
                        sample[j],
                        f"{class_dir}/{str(index).zfill(6)}.jpg"
                    )
            generator.to("cpu")
            utils.save_image(
                sample,
                f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}-overview.jpg",
                nrow=args.overview_n_columns
            )



def generate_gaussian_noise(args, device):

    noise_num = args.noise_num

    n_dim = 8*8 * 3   # 3 channels

    # transform = torch.rand([noise_num, n_dim, n_dim]) + torch.eye(n_dim).unsqueeze(0).repeat(noise_num, 1, 1)

    # transform_list = []
    # for i in range(noise_num):
    #     transform = orthogo_tensor(n_dim, n_dim)
    #     transform_list.append(transform.unsqueeze(0))
    # transform = torch.cat(transform_list, dim=0)

    # transform = torch.cos(torch.randn([noise_num, n_dim, n_dim]))
    # transform = transform / transform.norm(dim=0, keepdim=True)

    # transform = transform.to(device)

    upsamp = nn.Upsample(size=(args.image_resolution, args.image_resolution), mode='bilinear')

    # n_dim = args.image_resolution*args.image_resolution * 3   # 3 channels

    # normed_n_mean = train_distribution_diversity(
    #     n_distribution=noise_num, n_dim=n_dim, max_iters=500)
    zeros = torch.zeros(noise_num, n_dim)
    normed_n_mean = torch.normal(mean=zeros, std=1.0)
    latent_noise_mean = normed_n_mean.detach()
    latent_noise_std = 0.5

    with torch.no_grad():
        for noise_i in tqdm(range(args.noise_num)):
            iters = args.sample // args.batch_size

            train = "train"
            class_dir = f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}"

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            for batch_i in tqdm(range(iters)):
                latent_noise_mean_i = latent_noise_mean[noise_i].repeat(args.batch_size, 1)
                sample = torch.normal(mean=latent_noise_mean_i, std=latent_noise_std)
                # sample = sample.reshape(args.batch_size, 3, args.image_resolution, args.image_resolution)
                sample = sample.to(device)
                # sample = torch.mm(sample, transform[noise_i])
                # sample = sample / sample.norm(dim=1, keepdim=True)
                # max_value, _ = sample.max(dim=1, keepdim=True)
                # min_value, _ = sample.min(dim=1, keepdim=True)
                # sample = (sample - min_value) / (max_value - min_value)

                sample = sample.reshape(args.batch_size, 3, 8, 8)
                sample = upsamp(sample)

                for _ in range(3):
                    data_channel = sample[:,_,:,:]
                    # data_channel = (data_channel - data_channel.mean()) / data_channel.max()
                    # logging.debug(f"data std: {torch.std(data[:,_,:,:])} mean: {torch.mean(data[:,_,:,:])}")
                    print(f"data std: {torch.std(data_channel)} mean: {torch.mean(data_channel)}")

                for j in range(args.batch_size):
                    index = batch_i * args.batch_size + j 
                    utils.save_image(
                        sample[j],
                        f"{class_dir}/{str(index).zfill(6)}.jpg"
                    )
            utils.save_image(
                sample,
                f"{args.root_path}/{args.generate_dataset}/{train}/Distribution-{noise_i}-overview.jpg",
                nrow=args.overview_n_columns
            )








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the generator")


    parser.add_argument(
        "--model", type=str, default="style_GAN_v2_G", help="output image size of the generator"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10", help="Original dataset"
    )
    parser.add_argument(
        "--generate_dataset", type=str, default="style_GAN_init", help="output image size of the generator"
    )
    parser.add_argument(
        "--gpu_index", type=int, default=0, help="output image size of the generator"
    )
    parser.add_argument(
        "--fedaux", type=bool, default=False, help="output image size of the generator"
    )
    parser.add_argument(
        "--VHL", type=bool, default=True, help="output image size of the generator"
    )
    parser.add_argument(
        "--VHL_label_style", type=bool, default=True, help=""
    )
    parser.add_argument(
        "--fed_moon", type=bool, default=False, help=""
    )
    parser.add_argument(
        "--gate_layer", type=bool, default=False, help=""
    )


    parser.add_argument(
        "--root_path", type=str, default="./dataset", help="output image size of the generator"
    )

    parser.add_argument(
        "--image_resolution", type=int, default=32, help="output image size of the generator"
    )
    parser.add_argument(
        "--noise_num", type=int, default=10, help=""
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help=""
    )
    parser.add_argument(
        "--overview_n_columns", type=int, default=10, help=""
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="number of samples to be generated for each label",
    )
    parser.add_argument(
        "--style_gan_style_dim", type=int, default=64, help=""
    )
    parser.add_argument(
        "--style_gan_n_mlp", type=int, default=1, help=""
    )
    parser.add_argument("--package", type=bool, default=True, help="")
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--style_gan_cmul",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--style_gan_sample_z_mean",
        type=float,
        default=0.5,
        help="",
    )
    parser.add_argument(
        "--style_gan_sample_z_std",
        type=float,
        default=0.3,
        help="",
    )
    parser.add_argument(
        "--vae_decoder_z_dim",
        type=int,
        default=8,
        help="",
    )
    parser.add_argument(
        "--vae_decoder_ngf",
        type=int,
        default=64,
        help="",
    )



    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu_index) if torch.cuda.is_available() else "cpu")

    time_table = {}
    time_now = time.time()
    print("Creating model")

    # g_ema.load_state_dict(checkpoint["g_ema"])

    print("generating images")
    if args.model == "cifar_conv_decoder":
        generate_from_vae_decoder(args, device)
    elif args.model == "style_GAN_v2_G":
        generate(args, device)
    elif args.model == "Gaussian_Noise":
        generate_gaussian_noise(args, device)
    else:
        raise NotImplementedError

    time_table['generate_images'] = time.time() - time_now
    time_now = time.time()
    print(time_table)











