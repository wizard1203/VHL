import os
import argparse
import time
import math
import logging

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler




def load_iid_mnist(dataset, data_dir, partition_method, 
        partition_alpha, client_number, batch_size, rank=0):

    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)

    image_size = 28
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MNIST_MEAN , std=MNIST_STD),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MNIST_MEAN , std=MNIST_STD),
        ])

    train_dataset = MNIST(root=data_dir, train=True, \
                            transform=train_transform, download=False)

    test_dataset = MNIST(root=data_dir, train=False, \
                            transform=test_transform, download=False)

    train_sampler = None
    test_sampler = None
    shuffle = True
    if client_number > 1:
        train_sampler = data.distributed.DistributedSampler(
            train_dataset, num_replicas=client_number, rank=rank)
        train_sampler.set_epoch(0)
        shuffle = False

        # Note that test_sampler is for distributed testing to accelerate training
        test_sampler = data.distributed.DistributedSampler(
            test_dataset, num_replicas=client_number, rank=rank)
        train_sampler.set_epoch(0)


    train_data_global = data.DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4)
    test_data_global = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    train_sampler = train_sampler
    train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4, sampler=train_sampler)
    test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4, sampler=test_sampler)


    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_num = 10

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    # For local client, every dataloader is the same
    for client_index in range(client_number):
        train_data_local_dict[client_index] = train_dl
        test_data_local_dict[client_index] = test_dl
        data_local_num_dict[client_index] = train_data_num // client_number
        logging.info("client_index = %d, local_sample_number = %d" % (client_index, train_data_num))

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num








