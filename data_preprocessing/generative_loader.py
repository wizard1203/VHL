import logging
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data_preprocessing.utils.imbalance_data import ImbalancedDatasetSampler

from data_preprocessing.generative.datasets import GenerativeDataset
from data_preprocessing.generative.datasets import data_transforms_generative

from data_preprocessing.loader import Data_Loader


GENERATIVE_DATASET_LIST = ["style_GAN_init", "style_GAN_init_32_c100", "style_GAN_init_32_c62", "style_GAN_init_64_c200",
                        "Gaussian_Noise", "cifar_conv_decoder"]

class Generative_Data_Loader(Data_Loader):

    full_data_obj_dict = {
        "style_GAN_init": GenerativeDataset,
        "style_GAN_init_32_c100": GenerativeDataset,
        "style_GAN_init_32_c62": GenerativeDataset,
        "style_GAN_init_64_c200": GenerativeDataset,
        "Gaussian_Noise": GenerativeDataset,
        "cifar_conv_decoder": GenerativeDataset,
    } 

    sub_data_obj_dict = {
        "style_GAN_init": GenerativeDataset,
        "style_GAN_init_32_c100": GenerativeDataset,
        "style_GAN_init_32_c62": GenerativeDataset,
        "style_GAN_init_64_c200": GenerativeDataset,
        "Gaussian_Noise": GenerativeDataset,
        "cifar_conv_decoder": GenerativeDataset,
    } 

    transform_dict = {
        "style_GAN_init": data_transforms_generative,
        "style_GAN_init_32_c100": data_transforms_generative,
        "style_GAN_init_32_c62": data_transforms_generative,
        "style_GAN_init_64_c200": data_transforms_generative,
        "Gaussian_Noise": data_transforms_generative,
        "cifar_conv_decoder": data_transforms_generative,
    }

    num_classes_dict = {
        "style_GAN_init": 10,
        "style_GAN_init_32_c100": 100,
        "style_GAN_init_32_c62": 62,
        "style_GAN_init_64_c200": 200,
        "Gaussian_Noise": 10,
        "cifar_conv_decoder": 10,
    }


    image_resolution_dict = {
        "style_GAN_init": 32,
        "style_GAN_init_32_c100": 32,
        "style_GAN_init_32_c62": 32,
        "style_GAN_init_64_c200": 64,
        "Gaussian_Noise": 32,
        "cifar_conv_decoder": 32,
    }


    def __init__(self, args=None, process_id=0, mode="centralized", task="centralized",
                data_efficient_load=True, dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="iid", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default", other_params={}):
        super().__init__(args=args, process_id=process_id, mode=mode, task=task,
                data_efficient_load=data_efficient_load, dirichlet_balance=dirichlet_balance, dirichlet_min_p=dirichlet_min_p,
                dataset=dataset, datadir=datadir, partition_method=partition_method, partition_alpha=partition_alpha, client_number=client_number,
                batch_size=batch_size, num_workers=num_workers,
                data_sampler=data_sampler,
                resize=resize, augmentation=augmentation, other_params=other_params)




    def init_dataset_obj(self):
        self.full_data_obj = Generative_Data_Loader.full_data_obj_dict[self.dataset]
        self.sub_data_obj = Generative_Data_Loader.sub_data_obj_dict[self.dataset]
        logging.info(f"dataset augmentation: {self.augmentation}, resize: {self.resize}")
        self.get_transform_func = Generative_Data_Loader.transform_dict[self.dataset]
        self.class_num = Generative_Data_Loader.num_classes_dict[self.dataset]
        self.image_resolution = Generative_Data_Loader.image_resolution_dict[self.dataset]




    def load_full_data(self):
        # For cifar10, cifar100, SVHN, FMNIST
        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "full_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        # train_ds = GenerativeDataset(args, dataset_name=dataset_name, datadir=datadir,
        #         dataidxs=dataidxs,
        #         train=True, transform=train_transform, target_transform=None,
        #         load_in_memory=load_in_memory,
        #         image_resolution=image_resolution)

        train_ds = self.full_data_obj(self.args, dataset_name=self.dataset, datadir=self.datadir,
                dataidxs=None,
                train=True, transform=train_transform, target_transform=None,
                load_in_memory=False,
                image_resolution=self.image_resolution)

        test_ds = []
        # X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
        # X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets

        return train_ds, test_ds


    def load_centralized_data(self):
        self.train_ds, self.test_ds = self.load_full_data()
        self.train_data_num = len(self.train_ds)
        self.test_data_num = len(self.test_ds)
        self.train_dl, self.test_dl = self.get_dataloader(
                self.train_ds, self.test_ds,
                shuffle=True, drop_last=True, train_sampler=None, num_workers=self.num_workers)















