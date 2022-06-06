import logging
import random
import math
import functools

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    SVHN,
    FashionMNIST,
    MNIST,
)


from .cifar10.datasets import CIFAR10_truncated_WO_reload
from .cifar100.datasets import CIFAR100_truncated_WO_reload
from .SVHN.datasets import SVHN_truncated_WO_reload
from .TinyImageNet.datasets import TinyImageNet, TinyImageNet_truncated
from .FashionMNIST.datasets import FashionMNIST_truncated_WO_reload
from .MNIST.datasets import MNIST_truncated_WO_reload
from .FederatedEMNIST.datasets import FEMNIST_truncated_WO_reload
from .FederatedEMNIST.datasets import FEMNIST

from .cifar10.datasets import data_transforms_cifar10
from .cifar100.datasets import data_transforms_cifar100
from .SVHN.datasets import data_transforms_SVHN
from .TinyImageNet.datasets import data_transforms_TinyImageNet
from .FashionMNIST.datasets import data_transforms_fmnist
from .MNIST.datasets import data_transforms_mnist
from .FederatedEMNIST.datasets import data_transforms_femnist


from data_preprocessing.utils.stats import record_net_data_stats

from data_preprocessing.utils.imbalance_data import ImbalancedDatasetSampler

from data_preprocessing.utils.transform_utils import TwoCropTransform


NORMAL_DATASET_LIST = ["cifar10", "cifar100", "SVHN",
                        "mnist", "fmnist", "femnist-digit", "Tiny-ImageNet-200"]



class Data_Loader(object):

    full_data_obj_dict = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "SVHN": SVHN,
        "mnist": MNIST, 
        "fmnist": FashionMNIST,
        'femnist-digit': FEMNIST,
        "Tiny-ImageNet-200": TinyImageNet,
    } 
    sub_data_obj_dict = {
        "cifar10": CIFAR10_truncated_WO_reload,
        "cifar100": CIFAR100_truncated_WO_reload,
        "SVHN": SVHN_truncated_WO_reload,
        "mnist": MNIST_truncated_WO_reload, 
        "fmnist": FashionMNIST_truncated_WO_reload,
        'femnist-digit': FEMNIST_truncated_WO_reload,
        "Tiny-ImageNet-200": TinyImageNet_truncated,
    } 

    transform_dict = {
        "cifar10": data_transforms_cifar10,
        "cifar100": data_transforms_cifar100,
        "SVHN": data_transforms_SVHN,
        "mnist": data_transforms_mnist,
        "fmnist": data_transforms_fmnist,
        'femnist-digit': data_transforms_femnist,
        "Tiny-ImageNet-200": data_transforms_TinyImageNet,
    }

    num_classes_dict = {
        "cifar10": 10,
        "cifar100": 100,
        "SVHN": 10,
        "mnist": 10,
        "fmnist": 10,
        'femnist-digit': 10,
        "Tiny-ImageNet-200": 200,
    }


    image_resolution_dict = {
        "cifar10": 32,
        "cifar100": 32,
        "SVHN": 32,
        "mnist": 28,
        "fmnist": 32,
        'femnist-digit': 28,
        "Tiny-ImageNet-200": 64,
    }


    def __init__(self, args=None, process_id=0, mode="centralized", task="centralized",
                data_efficient_load=True, dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="iid", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default", other_params={}):

        # less use this.
        self.args = args

        # For partition
        self.process_id = process_id
        self.mode = mode
        self.task = task
        self.data_efficient_load = data_efficient_load # Loading mode, for standalone usage.
        self.dirichlet_balance = dirichlet_balance
        self.dirichlet_min_p = dirichlet_min_p

        self.dataset = dataset
        self.datadir = datadir
        self.partition_method = partition_method
        self.partition_alpha = partition_alpha
        self.client_number = client_number
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_sampler = data_sampler

        self.augmentation = augmentation
        self.other_params = other_params

        # For image
        self.resize = resize

        self.init_dataset_obj()


    def init_dataset_obj(self):
        self.full_data_obj = Data_Loader.full_data_obj_dict[self.dataset]
        self.sub_data_obj = Data_Loader.sub_data_obj_dict[self.dataset]
        logging.info(f"dataset augmentation: {self.augmentation}, resize: {self.resize}")
        self.get_transform_func = Data_Loader.transform_dict[self.dataset]
        self.class_num = Data_Loader.num_classes_dict[self.dataset]
        self.image_resolution = Data_Loader.image_resolution_dict[self.dataset]

        # client_index = process_id - 1. (in PS mode)
        # client_index = process_id. (in distributed mode of distributed training)

    def get_transform(self, resize, augmentation, dataset_type="full_dataset", image_resolution=32):
        MEAN, STD, train_transform, test_transform = \
            self.get_transform_func(
                resize=resize, augmentation=augmentation, dataset_type=dataset_type, image_resolution=image_resolution)
        # if self.args.Contrastive == "SimCLR":
        if self.args.TwoCropTransform:
            train_transform = TwoCropTransform(train_transform)
        return MEAN, STD, train_transform, test_transform

    def load_data(self):
        # Refered methods can be re-implemented by other data loader.
        if self.task == "federated":
            if self.mode == "distributed":
                self.federated_distributed_split()
            elif self.mode == "standalone":
                self.federated_standalone_split()
            else:
                raise NotImplementedError
        elif self.task == "distributed":
            if self.mode == "PS-distributed":
                self.rank = max(0, self.process_id - 1)
                self.distributed_PS_split()
            elif self.mode == "hvd-distributed":
                #TODO
                pass
            elif self.mode == "Gossip-distributed":
                self.rank = self.process_id
                self.distributed_Gossip_split()
            elif self.mode == "standalone":
                self.rank = self.process_id
                self.distributed_standalone_split()
            else:
                raise NotImplementedError
        elif self.task == "centralized":
            if self.mode == "centralized":
                self.load_centralized_data()
            else:
                raise NotImplementedError
            # self.train_ds = train_ds
            # self.test_ds = test_ds
            return self.train_dl, self.test_dl, self.train_data_num, self.test_data_num, self.class_num, self.other_params
        else:
            raise NotImplementedError
        self.other_params["traindata_cls_counts"] = self.traindata_cls_counts
        return self.train_data_num, self.test_data_num, self.train_data_global, self.test_data_global, \
            self.data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict, self.class_num, self.other_params




    def load_full_data(self):
        # For cifar10, cifar100, SVHN, FMNIST
        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "full_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        if self.dataset == "SVHN":
            train_ds = self.full_data_obj(self.datadir,  "train", download=False, transform=train_transform, target_transform=None)
            test_ds = self.full_data_obj(self.datadir,  "test", download=False, transform=test_transform, target_transform=None)
        elif self.dataset == "Tiny-ImageNet-200":
            train_ds = self.full_data_obj(self.datadir,  train=True, transform=train_transform, alpha=None)
            test_ds = self.full_data_obj(self.datadir,  train=False, transform=test_transform, alpha=None)
        else:
            train_ds = self.full_data_obj(self.datadir,  train=True, download=False, transform=train_transform)
            test_ds = self.full_data_obj(self.datadir,  train=False, download=False, transform=test_transform)

        # X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
        # X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets

        return train_ds, test_ds


    def load_sub_data(self, client_index, train_ds, test_ds):
        # Maybe only ``federated`` needs this.
        dataidxs = self.net_dataidx_map[client_index]
        local_data_num = len(dataidxs)

        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "sub_dataset", self.image_resolution)
        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        train_ds_local = self.sub_data_obj(self.datadir, dataidxs=dataidxs, train=True, transform=train_transform,
                full_dataset=train_ds)
        test_ds_local = self.sub_data_obj(self.datadir, train=False, transform=test_transform,
                        full_dataset=test_ds)
        return train_ds_local, test_ds_local, local_data_num


    def get_train_sampler(self, train_ds, rank, distributed=False):
        if distributed:
            train_sampler = data.distributed.DistributedSampler(
                train_ds, num_replicas=self.client_number, rank=rank)
            train_sampler.set_epoch(0)
        else:
            if self.data_sampler in ["imbalance", "decay_imb"]:
                train_sampler = ImbalancedDatasetSampler(self.args, train_ds, class_num=self.class_num)
            else:
                train_sampler = None
        return train_sampler


    def get_dataloader(self, train_ds, test_ds, shuffle=True, drop_last=False, train_sampler=None, num_workers=1):
        logging.info(f"shuffle: {shuffle}, drop_last:{drop_last}, train_sampler:{train_sampler} ")
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=shuffle,
                                drop_last=drop_last, sampler=train_sampler, num_workers=num_workers)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=False,
                                drop_last=False, num_workers=num_workers)
        return train_dl, test_dl


    def get_y_train_np(self, train_ds):
        if self.dataset in ["fmnist"]:
            y_train = train_ds.targets.data
        elif self.dataset in ["SVHN"]:
            y_train = train_ds.labels
        else:
            y_train = train_ds.targets
        y_train_np = np.array(y_train)
        return y_train_np


    # federated loading 
    def federated_distributed_split(self):
        raise NotImplementedError


    def federated_standalone_split(self):
        # For cifar10, cifar100, SVHN, FMNIST
        train_ds, test_ds = self.load_full_data()
        # y_train = train_ds.targets
        # y_train_np = np.array(y_train)
        y_train_np = self.get_y_train_np(train_ds)
        # class_num = len(np.unique(y_train))
        self.train_data_num = y_train_np.shape[0]
        self.net_dataidx_map, self.traindata_cls_counts = self.partition_data(y_train_np, self.train_data_num)
        logging.info("traindata_cls_counts = " + str(self.traindata_cls_counts))
        self.train_data_num = sum([len(self.net_dataidx_map[r]) for r in range(self.client_number)])

        self.train_data_global, self.test_data_global = self.get_dataloader(
                train_ds, test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)
        logging.info("train_dl_global number = " + str(len(self.train_data_global)))
        logging.info("test_dl_global number = " + str(len(self.test_data_global)))
        self.test_data_num = len(self.test_data_global)

        self.data_local_num_dict = dict()
        self.train_data_local_dict = dict()
        self.test_data_local_dict = dict()

        for client_index in range(self.client_number):
            train_ds_local, test_ds_local, local_data_num = self.load_sub_data(client_index, train_ds, test_ds)
            self.data_local_num_dict[client_index] = local_data_num
            logging.info("client_index = %d, local_sample_number = %d" % (client_index, local_data_num))

            train_sampler = self.get_train_sampler(train_ds_local, rank=client_index, distributed=False)
            shuffle = train_sampler is None

            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = self.get_dataloader(
                    train_ds_local, test_ds_local,
                    shuffle=shuffle, drop_last=False, train_sampler=train_sampler, num_workers=self.num_workers)
            logging.info("client_index = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_index, len(train_data_local), len(test_data_local)))
            self.train_data_local_dict[client_index] = train_data_local
            self.test_data_local_dict[client_index] = test_data_local


    # Distributed loading 
    def distributed_PS_split(self):

        train_ds, test_ds = self.load_full_data()

        self.train_data_global, self.test_data_global = self.get_dataloader(
                train_ds, test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)
        logging.info("train_dl_global number = " + str(len(self.train_data_global)))
        logging.info("test_dl_global number = " + str(len(self.test_data_global)))

        self.traindata_cls_counts = None

        self.train_data_num = len(train_ds)
        self.test_data_num = len(test_ds)

        self.data_local_num_dict = dict()
        self.train_data_local_dict = dict()
        self.test_data_local_dict = dict()

        for client_index in range(self.client_number):
            if client_index == self.rank:

                train_sampler = None
                if self.client_number > 1:

                    train_sampler = self.get_train_sampler(train_ds, rank=self.rank, distributed=True)
                    train_sampler.set_epoch(0)
                shuffle = train_sampler is None

                train_data_local, test_data_local = self.get_dataloader(
                        train_ds, test_ds,
                        shuffle=shuffle, drop_last=False, train_sampler=train_sampler, num_workers=self.num_workers)

                self.train_data_local_dict[client_index] = train_data_local
                self.test_data_local_dict[client_index] = test_data_local
                # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
                self.data_local_num_dict[client_index] = self.train_data_num // self.client_number
                logging.info("client_index = %d, local_sample_number = %d" % (client_index, self.train_data_num))
            else:
                # If algorithm uses this loader, raise errors.
                self.train_data_local_dict[client_index] = None
                self.test_data_local_dict[client_index] = None
                # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
                self.data_local_num_dict[client_index] = self.train_data_num // self.client_number
                logging.info("client_index = %d, local_sample_number = %d" % (client_index, self.train_data_num))


    def distributed_Gossip_split(self):

        train_ds, test_ds = self.load_full_data()

        self.train_data_global, self.test_data_global = self.get_dataloader(
                train_ds, test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)
        logging.info("train_dl_global number = " + str(len(self.train_data_global)))
        logging.info("test_dl_global number = " + str(len(self.test_data_global)))

        self.traindata_cls_counts = None

        self.train_data_num = len(train_ds)
        self.test_data_num = len(test_ds)

        self.data_local_num_dict = dict()
        self.train_data_local_dict = dict()
        self.test_data_local_dict = dict()

        for client_index in range(self.client_number):
            if client_index == self.rank:

                train_sampler = None
                if self.client_number > 1:

                    train_sampler = self.get_train_sampler(train_ds, rank=self.rank, distributed=True)
                    train_sampler.set_epoch(0)
                shuffle = train_sampler is None

                train_data_local, test_data_local = self.get_dataloader(
                        train_ds, test_ds,
                        shuffle=shuffle, drop_last=False, train_sampler=train_sampler, num_workers=self.num_workers)

                self.train_data_local_dict[client_index] = train_data_local
                self.test_data_local_dict[client_index] = test_data_local
                # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
                self.data_local_num_dict[client_index] = self.train_data_num // self.client_number
                logging.info("client_index = %d, local_sample_number = %d" % (client_index, self.train_data_num))
            else:
                # If algorithm uses this loader, raise errors.
                self.train_data_local_dict[client_index] = None
                self.test_data_local_dict[client_index] = None
                # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
                self.data_local_num_dict[client_index] = self.train_data_num // self.client_number
                logging.info("client_index = %d, local_sample_number = %d" % (client_index, self.train_data_num))


    # Distributed loading 
    def distributed_standalone_split(self):

        train_ds, test_ds = self.load_full_data()

        self.train_data_global, self.test_data_global = self.get_dataloader(
                train_ds, test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)
        logging.info("train_dl_global number = " + str(len(self.train_data_global)))
        logging.info("test_dl_global number = " + str(len(self.test_data_global)))

        self.traindata_cls_counts = None

        self.train_data_num = len(train_ds)
        self.test_data_num = len(test_ds)

        self.data_local_num_dict = dict()
        self.train_data_local_dict = dict()
        self.test_data_local_dict = dict()

        # Standalone version, cients directly read data from the global loader.
        for client_index in range(self.client_number):
            self.train_data_local_dict[client_index] = None
            self.test_data_local_dict[client_index] = None
            # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
            self.data_local_num_dict[client_index] = self.train_data_num // self.client_number
            logging.info("client_index = %d, local_sample_number = %d" % (client_index, self.train_data_num))




    # centralized loading
    def load_centralized_data(self):
        self.train_ds, self.test_ds = self.load_full_data()
        self.train_data_num = len(self.train_ds)
        self.test_data_num = len(self.test_ds)
        self.train_dl, self.test_dl = self.get_dataloader(
                self.train_ds, self.test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)






    def partition_data(self, y_train_np, train_data_num):

        if self.partition_method in ["homo", "iid"]:
            total_num = train_data_num
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.client_number)
            net_dataidx_map = {i: batch_idxs[i] for i in range(self.client_number)}

        elif self.partition_method == "hetero":
            min_size = 0
            K = self.class_num
            N = y_train_np.shape[0]
            logging.info("N = " + str(N))
            net_dataidx_map = {}

            while min_size < self.class_num:
                idx_batch = [[] for _ in range(self.client_number)]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_train_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.partition_alpha, self.client_number))
                    ## Balance
                    # used_p = np.array([len(idx_j) for idx_j in idx_batch])
                    # if used_p.sum() > 0:
                    #     used_p = used_p / used_p.sum()
                    # else:
                    #     used_p = np.array([1 for i in range(self.client_number)])
                    #     used_p = used_p / used_p.sum()
                    # used_p = 1 - used_p
                    # used_p = used_p / used_p.sum()
                    # # proportions = proportions + 0.5*used_p
                    # proportions = proportions + 5*used_p
                    # proportions = np.array([p * (len(idx_j) < N / self.client_number) for p, idx_j in zip(proportions, idx_batch)])
                    # proportions = np.array([p * ( p*5000 + len(idx_j) < N / self.client_number) for p, idx_j in zip(proportions, idx_batch)])
                    if self.dirichlet_balance:
                        argsort_proportions = np.argsort(proportions, axis=0)
                        if k != 0:
                            used_p = np.array([len(idx_j) for idx_j in idx_batch])
                            argsort_used_p = np.argsort(used_p, axis=0)
                            inv_argsort_proportions = argsort_proportions[::-1]
                            # print(used_p)
                            # print(argsort_used_p)
                            # proportions = np.random.random(self.client_number)
                            proportions[argsort_used_p] = proportions[inv_argsort_proportions]
                            # print(np.argsort(proportions, axis=0))
                    else:
                        proportions = np.array([p * (len(idx_j) < N / self.client_number) for p, idx_j in zip(proportions, idx_batch)])

                    ## set a min value to smooth, avoid too much zero samples of some classes.
                    if self.dirichlet_min_p is not None:
                        proportions += float(self.dirichlet_min_p)
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

        elif self.partition_method == "LDA_v2":
            """
            modified from
            https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/datasets/partition_data.py#L196-L274
            """ 
            num_classes = self.class_num
            num_indices = y_train_np.shape[0]
            n_workers = self.client_number
            indices = np.array([x for x in range(0, len(y_train_np))])

            random_state = np.random.RandomState(self.args.seed)
            list_of_indices = build_non_iid_by_dirichlet(
                random_state=random_state,
                indices2targets=np.array(
                    [
                        (idx, target)
                        for idx, target in enumerate(y_train_np)
                        if idx in indices
                    ]
                ),
                non_iid_alpha=self.partition_alpha,
                num_classes=num_classes,
                num_indices=num_indices,
                n_workers=n_workers,
            )
            # indices = functools.reduce(lambda a, b: a + b, list_of_indices)
            # print(f"indices: {indices}")
            # print(f"list_of_indices: {list_of_indices}")
            # print(f"indices: {len(indices)}")
            print(f"list_of_indices: {len(list_of_indices)}")
            net_dataidx_map = {i: list_of_indices[i] for i in range(self.client_number)}

        # refer to https://github.com/Xtra-Computing/NIID-Bench/blob/main/utils.py
        elif self.partition_method > "noniid-#label0" and self.partition_method <= "noniid-#label9":
            num = eval(self.partition_method[13:])
            if self.dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
                num = 1
                K = 2
            else:
                K = self.class_num
            if num == 10:
                net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(self.client_number)}
                for i in range(10):
                    idx_k = np.where(y_train_np==i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.client_number)
                    for j in range(self.client_number):
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
            else:
                times=[0 for i in range(10)]
                contain=[]
                for i in range(self.client_number):
                    current=[i%K]
                    times[i%K]+=1
                    j=1
                    while (j<num):
                        ind=random.randint(0,K-1)
                        if (ind not in current):
                            j=j+1
                            current.append(ind)
                            times[ind]+=1
                    contain.append(current)
                net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(self.client_number)}
                for i in range(K):
                    idx_k = np.where(y_train_np==i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k,times[i])
                    ids=0
                    for j in range(self.client_number):
                        if i in contain[j]:
                            net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                            ids+=1
        elif self.partition_method == "long-tail":
            if self.client_number == 10 or self.client_number == 100:
                pass
            else:
                raise NotImplementedError
            
            # There are  self.client_number // self.class_num clients share the \alpha proportion of data of one class
            main_prop = self.partition_alpha / (self.client_number // self.class_num)

            # There are (self.client_number - self.client_number // self.class_num) clients share the tail of one class
            tail_prop = (1 - main_prop) / (self.client_number - self.client_number // self.class_num)

            net_dataidx_map = {}
            # for each class in the dataset
            K = self.class_num
            idx_batch = [[] for _ in range(self.client_number)]
            for k in range(K):
                idx_k = np.where(y_train_np == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.array([ tail_prop for _ in range(self.client_number)])
                main_clients = np.array([ k + i*K for i in range(self.client_number // K)])
                proportions[main_clients] = main_prop
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]


        elif self.partition_method == "hetero-fix":
            pass
            # dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
            # net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

        if self.partition_method == "hetero-fix":
            pass
            # distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
            # traindata_cls_counts = read_data_distribution(distribution_file_path)
        else:
            traindata_cls_counts = record_net_data_stats(y_train_np, net_dataidx_map)

        return net_dataidx_map, traindata_cls_counts





"""
modified from
https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/datasets/partition_data.py#L196-L274
"""


def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch











