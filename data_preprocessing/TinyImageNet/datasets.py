import os, sys
import os.path
import logging
import numpy as np

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from PIL import Image

import torch
import torchvision.transforms as transforms

from data_preprocessing.utils.utils import Cutout

from fedml_core.non_iid_partition.noniid_partition import record_data_stats, \
    non_iid_partition_with_dirichlet_distribution



def data_transforms_TinyImageNet(resize=64, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    if augmentation == "default":
        if resize is 64:
            train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomCrop(resize, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        train_transform.transforms.append(Cutout(16))
    else:
        if resize is 64:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])


    return IMAGENET_MEAN, IMAGENET_STD, train_transform, test_transform




class TinyImageNet(Dataset):
    def __init__(self, data_dir, dataidxs=None, train=True, transform=None,
                    client_num=100, alpha=None):

        self.dataidxs = dataidxs
        self.client_num = client_num
        self.Train = train
        self.root_dir = data_dir
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self.alpha = alpha
        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

        self.initial_local_data()
        self.class_num = 200

    def initial_local_data(self):
        if self.dataidxs == None:
            self.local_data = self.all_data
        elif type(self.dataidxs) == int:
            if self.alpha is not None:
                # self.local_data = self.all_data[self.net_dataidx_map[self.dataidxs]]
                # self.local_data = list(np.array(self.all_data)[self.net_dataidx_map[self.dataidxs]])
                self.local_data = [self.all_data[idx] for idx in self.net_dataidx_map[self.dataidxs] ]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


    def get_local_data(self):
        return self.local_data

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train, data_dir, class_to_idx):
        images = []

        data_local_num_dict = dict()
        net_dataidx_map = dict()
        sum_temp = 0
        data_dir = os.path.expanduser(data_dir)

        if Train:
            img_root_dir = data_dir
            list_of_dirs = [target for target in class_to_idx.keys()]
        else:
            img_root_dir = data_dir
            list_of_dirs = ["images"]

        for target in list_of_dirs:
            dirs = os.path.join(img_root_dir, target)
            if not os.path.isdir(dirs):
                continue

            target_num = 0
            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, class_to_idx[target])
                        else:
                            item = (path, class_to_idx[self.val_img_to_class[fname]])
                        images.append(item)
                        target_num += 1
            if Train:
                net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
                data_local_num_dict[class_to_idx[target]] = target_num
                sum_temp += target_num

                assert len(images) == sum_temp
        # return np.array(images), data_local_num_dict, net_dataidx_map
        return images, data_local_num_dict, net_dataidx_map


    def _make_dataset_with_dirichlet_sampling(self, Train, data_dir, class_to_idx, client_num, alpha):
        assert alpha > 0
        images = []

        data_local_num_dict = dict()
        net_dataidx_map = dict()
        sum_temp = 0
        data_dir = os.path.expanduser(data_dir)

        if Train:
            img_root_dir = data_dir
            list_of_dirs = [target for target in class_to_idx.keys()]
        else:
            img_root_dir = data_dir
            list_of_dirs = ["images"]

        label_list = []     # Used for dirichlet sampling
        for target in list_of_dirs:
            dirs = os.path.join(img_root_dir, target)
            if not os.path.isdir(dirs):
                continue

            target_num = 0
            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            label = class_to_idx[target]
                            item = (path, label)
                        else:
                            label = class_to_idx[self.val_img_to_class[fname]]
                            item = (path, label)
                        # logging.debug(f"label: {label}, target: {target}, ")
                        label_list.append(label)
                        images.append(item)
                        target_num += 1
            if Train:
                net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
                data_local_num_dict[class_to_idx[target]] = target_num
                sum_temp += target_num

        if Train:
            self.label_list = np.array(label_list)
            # net_dataidx_map = non_iid_partition_with_dirichlet_distribution(
            #     label_list=self.label_list, client_num=client_num, classes=100, alpha=alpha)
            net_dataidx_map = non_iid_partition_with_dirichlet_distribution(
                label_list=self.label_list, client_num=client_num, classes=200, alpha=alpha)

            assert len(images) == sum_temp
        # return np.array(images), data_local_num_dict, net_dataidx_map
        return images, data_local_num_dict, net_dataidx_map


    def __getdatasets__(self):

        if self.Train:
            data_dir = self.train_dir
        else:
            data_dir = self.val_dir

        if self.alpha is not None:
            all_data, data_local_num_dict, net_dataidx_map = self._make_dataset_with_dirichlet_sampling(
                self.Train, data_dir, self.class_to_tgt_idx, self.client_num, alpha=self.alpha
            )
        else:
            all_data, data_local_num_dict, net_dataidx_map = self._make_dataset(self.Train, data_dir, self.class_to_tgt_idx)


        return all_data, data_local_num_dict, net_dataidx_map



    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.local_data[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        # logging.debug(f"img_path: {img_path}, sample: {type(sample)}, sample.shape: {sample.shape}, tgt: {tgt}" )

        return sample, tgt




class TinyImageNet_truncated(Dataset):
    def __init__(self, full_dataset: TinyImageNet, dataidxs, net_dataidx_map, 
                train=True, transform=None,
                client_num=100, alpha=None):

        self.dataidxs = dataidxs
        self.client_num = client_num
        self.train = train
        self.transform = transform
        self.net_dataidx_map = net_dataidx_map
        self.all_data = full_dataset.get_local_data()
        self.alpha = alpha
        self.initial_local_data()


    def initial_local_data(self):
        if self.dataidxs == None:
            self.local_data = self.all_data
        elif type(self.dataidxs) == int:
            if self.alpha is not None:
                # self.local_data = self.all_data[self.net_dataidx_map[self.dataidxs]]
                # self.local_data = list(np.array(self.all_data)[self.net_dataidx_map[self.dataidxs]])
                self.local_data = [self.all_data[idx] for idx in self.net_dataidx_map[self.dataidxs] ]
            else:
                raise NotImplementedError
                # (begin, end) = self.net_dataidx_map[self.dataidxs]
                # self.local_data = self.all_data[begin: end]
        else:
            # This is only suitable when not do dirichlet sampling
            raise NotImplementedError


    def __len__(self):
        return len(self.local_data)


    def __getitem__(self, idx):
        img_path, tgt = self.local_data[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        # logging.debug(f"img_path: {img_path}, sample: {type(sample)}, sample.shape: {sample.shape}, tgt: {tgt}" )

        return sample, tgt











class TinyImageNet_subset(TinyImageNet):
    def __init__(self, data_dir, dataidxs=None, train=True, transform=None,
                    client_num=100, alpha=None, max_classes=200):
        self.max_classes = max_classes
        super().__init__(data_dir, dataidxs=dataidxs, train=train, transform=transform,
                    client_num=client_num, alpha=alpha)
        self.class_num = max_classes


    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)[:self.max_classes]
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        # self.len_dataset = num_images
        self.len_dataset = int(num_images * self.max_classes / 200)

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}


    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = int(len(list(self.val_img_to_class.keys())) * self.max_classes / 200)

        classes = sorted(list(set_of_classes))[:self.max_classes]
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}


    def _make_dataset(self, Train, data_dir, class_to_idx):
        images = []

        data_local_num_dict = dict()
        net_dataidx_map = dict()
        sum_temp = 0
        data_dir = os.path.expanduser(data_dir)

        if Train:
            img_root_dir = data_dir
            list_of_dirs = [target for target in class_to_idx.keys()]
        else:
            img_root_dir = data_dir
            list_of_dirs = ["images"]

        for target in list_of_dirs:
            dirs = os.path.join(img_root_dir, target)
            if not os.path.isdir(dirs):
                continue

            target_num = 0
            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, class_to_idx[target])
                        else:
                            if class_to_idx[self.val_img_to_class[fname]] > self.max_classes:
                                continue
                            item = (path, class_to_idx[self.val_img_to_class[fname]])
                        images.append(item)
                        target_num += 1
            if Train:
                net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
                data_local_num_dict[class_to_idx[target]] = target_num
                sum_temp += target_num

                assert len(images) == sum_temp
        # return np.array(images), data_local_num_dict, net_dataidx_map
        return images, data_local_num_dict, net_dataidx_map



    def __len__(self):
        return self.len_dataset



    def __getitem__(self, idx):
        img_path, tgt = self.local_data[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        # logging.debug(f"img_path: {img_path}, sample: {type(sample)}, sample.shape: {sample.shape}, tgt: {tgt}" )

        return sample, tgt




