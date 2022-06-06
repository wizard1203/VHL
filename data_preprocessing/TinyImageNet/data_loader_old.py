import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

# from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
# from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

from .datasets import TinyImageNet
from .datasets import TinyImageNet_truncated
from .datasets import TinyImageNet_subset


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_ImageNet(args, image_size=64):
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    # if args.dataset_resize:
    #     image_size = args.dataset_load_image_size
    # else:
    image_size = image_size

    # image_size = 64
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform




def partition_data(dataset, datadir, partition, n_nets, partition_alpha, args=None):

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
        train_dataset_global, test_dataset_global = None, None, None, None, None, None, None, None, 

    transform_train, transform_test = _data_transforms_ImageNet(args)
    train_dataset_global = TinyImageNet(data_dir=datadir,
                            dataidxs=None,
                            train=True,
                            transform=transform_train,
                            client_num=n_nets,
                            alpha=partition_alpha) 

    test_dataset_global = TinyImageNet(data_dir=datadir,
                            dataidxs=None,
                            train=False,
                            transform=transform_test,
                            client_num=n_nets,
                            alpha=None)
    class_num = 200

    net_dataidx_map = train_dataset_global.get_net_dataidx_map()

    traindata_cls_counts = record_net_data_stats(train_dataset_global.label_list, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
                    train_dataset_global, test_dataset_global


def get_dataloader(dataset_train, dataset_test, train_bs,
                    test_bs, dataidxs=None, net_dataidx_map=None, args=None):

    train_dl = data.DataLoader(dataset=dataset_train, batch_size=train_bs, shuffle=True, drop_last=False,
                        pin_memory=True, num_workers=4)
    if dataset_test is None:
        test_dl = None
    else:
        test_dl = data.DataLoader(dataset=dataset_test, batch_size=test_bs, shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=4)

    return train_dl, test_dl




def distributed_centralized_TinyImageNet_loader(dataset, data_dir, client_number, batch_size, rank=0, args=None):
    """
        Used for generating distributed dataloader for 
        accelerating centralized training 
    """

    train_bs=batch_size
    test_bs=batch_size

    transform_train, transform_test = _data_transforms_ImageNet(args)

    train_dataset_global = TinyImageNet(data_dir=data_dir,
                            dataidxs=None,
                            train=True,
                            transform=transform_train,
                            client_num=client_number) 

    test_dataset_global = TinyImageNet(data_dir=data_dir,
                            dataidxs=None,
                            train=False,
                            transform=transform_test,
                            client_num=client_number)
    class_num = 200

    train_sam = DistributedSampler(train_dataset_global, num_replicas=client_number, rank=rank)
    test_sam = DistributedSampler(test_dataset_global, num_replicas=client_number, rank=rank)

    # train_data_local = data.DataLoader(train_dataset_global, batch_size=train_bs , sampler=train_sam,
    #                     pin_memory=True, num_workers=4)
    train_global = data.DataLoader(train_dataset_global, batch_size=train_bs, shuffle=True, sampler=None,
                        pin_memory=True, num_workers=4)

    # test_data_local = data.DataLoader(test_dataset_global, batch_size=test_bs, sampler=test_sam,
    #                     pin_memory=True, num_workers=4)
    test_global = data.DataLoader(test_dataset_global, batch_size=test_bs, sampler=None,
                        pin_memory=True, num_workers=4)

    train_data_num = len(train_dataset_global)
    test_data_num = len(test_dataset_global)

    logging.info("len of train_dataset: {}".format(train_data_num))
    logging.info("len of test_dataset: {}".format(test_data_num))

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_index in range(client_number):
        # client_index, len(train_data_local), len(test_data_local)))
        data_local_num_dict[client_index] = train_data_num // client_number
        train_data_local_dict[client_index] = train_global
        test_data_local_dict[client_index] = test_global

    return train_data_num, test_data_num, train_global, test_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num



def load_centralized_Tiny_ImageNet_200(dataset, data_dir, batch_size, 
                  max_train_len=None, max_test_len=None,
                  args=None, max_classes=200, image_size=64, **kwargs):

    train_transform, test_transform = _data_transforms_ImageNet(args, image_size=image_size)

    if dataset == "Tiny-ImageNet-200":
        train_dataset_global = TinyImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=train_transform) 

        test_dataset_global = TinyImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=test_transform,
                                client_num=client_number)
        class_num = 200
    elif dataset == "Sub-Tiny-ImageNet-200":
        train_dataset_global = TinyImageNet_subset(data_dir=data_dir,
                                dataidxs=None,
                                train=True,
                                transform=train_transform,
                                max_classes=max_classes) 

        # Not support sub test dataset Now.
        test_dataset_global = TinyImageNet(data_dir=data_dir,
                                dataidxs=None,
                                train=False,
                                transform=test_transform)
        # test_dataset_global = TinyImageNet_subset(data_dir=data_dir,
        #                         dataidxs=None,
        #                         train=False,
        #                         transform=test_transform,
        #                         max_classes=max_classes)
        class_num = max_classes
    else:
        raise NotImplementedError


    shuffle = True

    train_dl = data.DataLoader(train_dataset_global, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4)
    test_dl = data.DataLoader(test_dataset_global, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    train_data_num = len(train_dataset_global)
    test_data_num = len(test_dataset_global)

    return train_dl, test_dl, train_data_num, test_data_num, class_num






def load_partition_data_TinyImageNet(dataset, data_dir, partition_method=None, partition_alpha=None, 
                                    client_number=200, batch_size=64, args=None):

    transform_train, transform_test = _data_transforms_ImageNet(args)

    train_dataset_global = TinyImageNet(data_dir=data_dir,
                            dataidxs=None,
                            train=True,
                            transform=transform_train,
                            client_num=client_number,
                            alpha=partition_alpha) 

    test_dataset_global = TinyImageNet(data_dir=data_dir,
                            dataidxs=None,
                            train=False,
                            transform=transform_test,
                            client_num=client_number,
                            alpha=None)
    class_num = 200

    net_dataidx_map = train_dataset_global.get_net_dataidx_map()

    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    train_data_num = len(train_dataset_global)
    test_data_num = len(test_dataset_global)
    # data_local_num_dict = train_dataset_global.get_data_local_num_dict()

    train_data_global, test_data_global = get_dataloader(
                                train_dataset_global, test_dataset_global,
                                train_bs=batch_size, test_bs=batch_size,
                                dataidxs=None, net_dataidx_map=None, args=None)

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_index in range(client_number):
        if partition_alpha is None:
            raise NotImplementedError("Not support other partition methods Now!")
        else:
            dataidxs = client_index

        train_dataset_local = TinyImageNet_truncated(
                train_dataset_global, dataidxs, net_dataidx_map, 
                train=True, transform=transform_train,
                client_num=client_number, alpha=partition_alpha)

        test_dataset_local = None

        train_data_local, _ = get_dataloader(train_dataset_local, test_dataset_local,
                                            train_bs=batch_size, test_bs=batch_size,
                                            dataidxs=None, net_dataidx_map=net_dataidx_map,
                                            args=args)

        data_local_num_dict[client_index] = len(train_dataset_local)
        train_data_local_dict[client_index] = train_data_local
        test_data_local_dict[client_index] = test_data_global


    logging.info("data_local_num_dict: %s" % data_local_num_dict)
    return train_data_num, test_data_num, train_data_global, test_data_global, \
            data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


if __name__ == '__main__':
    data_dir = '/home/datasets/imagenet/ILSVRC2012_dataset'

    client_number = 200
    train_data_num, test_data_num, train_data_global, test_data_global, \
    data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = \
        load_partition_data_TinyImageNet(None, data_dir,
                                     partition_method=None, partition_alpha=None, client_number=client_number,
                                     batch_size=10)

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    i = 0
    for data, label in train_data_global:
        print(data)
        print(label)
        i += 1
        if i > 5:
            break
    print("=============================\n")

    for client_index in range(client_number):
        i = 0
        for data, label in train_data_local_dict[client_index]:
            print(data)
            print(label)
            i += 1
            if i > 5:
                break
