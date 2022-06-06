
import logging
from re import T

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import torchvision



from .datasets import GenerativeDataset




def get_dataloader_Generative(datadir, train_bs, test_bs, dataidxs=None, args=None,
                        full_train_dataset=None, full_test_dataset=None,
                        dataset_name="style_GAN_init", 
                        resize_size=None, image_resolution=32, load_in_memory=False):

    transform_array = []
    if resize_size is not None:
        transform_array.append(
            torchvision.transforms.Resize(resize_size)
            # torchvision.transforms.Resize((96,96))
        )
        image_resolution = resize_size

    if args.generative_dataset_grayscale:
        transform_array.append(
            torchvision.transforms.Grayscale(num_output_channels=1)
        )
        GENERETIVE_MEAN = (0.5)
        GENERETIVE_STD = (0.25)
    else:
        GENERETIVE_MEAN = (0.5, 0.5, 0.5)
        GENERETIVE_STD = (0.25, 0.25, 0.25)

    # image_resolution = image_resolution

    transform_array += [
        torchvision.transforms.RandomCrop(image_resolution, padding=4),
        # torchvision.transforms.RandomResizedCrop(image_resolution, scale=(0.08, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(
        #     (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
        #     (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        # ),
        torchvision.transforms.Normalize(
            GENERETIVE_MEAN,
            GENERETIVE_STD,
        ),
    ]

    train_transform = torchvision.transforms.Compose(transform_array)

    train_ds = GenerativeDataset(args, dataset_name=dataset_name, datadir=datadir,
            dataidxs=dataidxs,
            train=True, transform=train_transform, target_transform=None,
            load_in_memory=load_in_memory,
            image_resolution=image_resolution)


    # Use drop_last for avoiding batch size unmatch
    # train_dl = data.DataLoader(train_ds, batch_size=train_bs, num_workers=args.data_load_num_workers,
    #                     shuffle=True, pin_memory=True)
    train_dl = data.DataLoader(train_ds, batch_size=train_bs, num_workers=args.data_load_num_workers,
                        shuffle=True, pin_memory=args.generative_dataset_pin_memory,
                        drop_last=True)

    test_ds = None

    test_dl = None


    return train_dl, test_dl, train_ds, test_ds












