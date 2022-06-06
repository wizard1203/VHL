import logging
import random
import os
import json

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .shakespeare.language_utils import word_to_indices, VOCAB_SIZE, \
    letter_to_index
from .shakespeare import utils

from .loader import Data_Loader


SHAKESPEARE_DATASET_LIST = ["shakespeare"]

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data



def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    return y_batch


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(process_x(batched_x)))
        batched_y = torch.from_numpy(np.asarray(process_y(batched_y)))
        batch_data.append((batched_x, batched_y))
    return batch_data


class Shakespeare_Data_Loader(Data_Loader):


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
        # self.output_dim = other_params["VOCAB_SIZE"]
        self.output_dim = VOCAB_SIZE
        self.class_num = self.output_dim


    def init_dataset_obj(self):
        self.image_resolution = None

        # self.full_data_obj = Generative_Data_Loader.full_data_obj_dict[self.dataset]
        # self.sub_data_obj = Generative_Data_Loader.sub_data_obj_dict[self.dataset]
        # logging.info(f"dataset augmentation: {self.augmentation}, resize: {self.resize}")
        # self.get_transform_func = Generative_Data_Loader.transform_dict[self.dataset]
        # self.class_num = Generative_Data_Loader.num_classes_dict[self.dataset]
        # self.image_resolution = Generative_Data_Loader.image_resolution_dict[self.dataset]




    def load_full_data(self):
        # train_path = "../../../data/shakespeare/train"
        # test_path = "../../../data/shakespeare/test"
        train_path = os.path.join(self.datadir, "train")
        test_path = os.path.join(self.datadir, "test")
        users, groups, train_data, test_data = read_data(train_path, test_path)

        if len(groups) == 0:
            groups = [None for _ in users]
        train_data_num = 0
        test_data_num = 0
        # client_index = 0

        all_train_data_x = []
        all_train_data_y = []
        all_test_data_x = []
        all_test_data_y = []

        for u, g in zip(users, groups):
            user_train_data_num = len(train_data[u]['x'])
            user_test_data_num = len(test_data[u]['x'])
            train_data_num += user_train_data_num
            test_data_num += user_test_data_num
            # train_data_local_num_dict[client_index] = user_train_data_num
            # transform to batches

            train_data_x = train_data[u]['x']
            train_data_y = train_data[u]['y']
            test_data_x = test_data[u]['x']
            test_data_y = test_data[u]['y']

            # loop through mini-batches
            # batch_data = list()
            # for i in range(0, len(data_x), batch_size):
            #     batched_x = data_x[i:i + batch_size]
            #     batched_y = data_y[i:i + batch_size]
            client_train_x = torch.from_numpy(np.asarray(process_x(train_data_x)))
            client_train_y = torch.from_numpy(np.asarray(process_y(train_data_y)))
            client_test_x = torch.from_numpy(np.asarray(process_x(test_data_x)))
            client_test_y = torch.from_numpy(np.asarray(process_y(test_data_y)))
            # batch_data.append((batched_x, batched_y))
            all_train_data_x.append(client_train_x)
            all_train_data_y.append(client_train_y)
            all_test_data_x.append(client_test_x)
            all_test_data_y.append(client_test_y)


        all_train_data_x = torch.cat(all_train_data_x)
        all_train_data_y = torch.cat(all_train_data_y)
        all_test_data_x = torch.cat(all_test_data_x)
        all_test_data_y = torch.cat(all_test_data_y)

        train_ds = data.TensorDataset(all_train_data_x,
                                    all_train_data_y)
        test_ds = data.TensorDataset(all_test_data_x,
                                    all_test_data_y)

        return train_ds, test_ds


    def load_centralized_data(self):
        self.train_ds, self.test_ds = self.load_full_data()
        self.train_data_num = len(self.train_ds)
        self.test_data_num = len(self.test_ds)
        self.train_dl, self.test_dl = self.get_dataloader(
                self.train_ds, self.test_ds,
                shuffle=True, drop_last=True, train_sampler=None, num_workers=self.num_workers)




    # federated loading 
    def federated_distributed_split(self):
        raise NotImplementedError


    def federated_standalone_split(self):
        raise NotImplementedError


    # Distributed loading 
    def distributed_standalone_split(self):
        raise NotImplementedError











