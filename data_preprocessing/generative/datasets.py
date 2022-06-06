import logging
import os, sys
import random

import numpy as np
import torch.utils.data as data
import torchvision
from PIL import Image
import torch
import torchvision.transforms as transforms


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


# LABELED_DATASET = (
#     "style_GAN_init",
#     "style_GAN_init_64"
# )



def data_transforms_generative(resize=None, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):

    train_transform = transforms.Compose([])
    test_transform = None

    if "grayscale" in augmentation:
        train_transform.transforms.append(
            torchvision.transforms.Grayscale(num_output_channels=1)
        )
        GENERETIVE_MEAN = (0.5)
        GENERETIVE_STD = (0.25)
    else:
        GENERETIVE_MEAN = (0.5, 0.5, 0.5)
        GENERETIVE_STD = (0.25, 0.25, 0.25)

    image_size = image_resolution
    if resize is not image_size:
        image_size = resize
        train_transform.transforms.append(transforms.Resize(resize))

    if "default" in augmentation:
        # pass
        train_transform.transforms.append(transforms.RandomCrop(image_size, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    elif augmentation == "no":
        pass
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(GENERETIVE_MEAN, GENERETIVE_STD))

    return GENERETIVE_MEAN, GENERETIVE_STD, train_transform, test_transform



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)



def find_classes(dir, labeled=True):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    if labeled:
        class_to_idx = {classes[i]: i for i in range(len(classes))}
    else:
        class_to_idx = {classes[i]: 0 for i in range(len(classes))}
    return classes, class_to_idx





def make_dataset(dir, class_to_idx, extensions, num_classes=1000, labeled=True):
    images = []

    data_local_num_dict = dict()
    data_local_num_dict[0] = 0
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)

    i_target = 0 
    for target in sorted(os.listdir(dir)):
        if not (i_target < num_classes):
            break
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        target_num = 0
        if labeled:
            label = class_to_idx[target]
        else:
            label = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    # item = (path, class_to_idx[target])
                    item = (path, label)
                    images.append(item)
                    target_num += 1

        # net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        # data_local_num_dict[class_to_idx[target]] = target_num

        if labeled:
            net_dataidx_map[label] = (sum_temp, sum_temp + target_num)
            data_local_num_dict[label] = target_num
        else:
            net_dataidx_map[label] = (0, sum_temp + target_num)
            data_local_num_dict[label] += target_num
        sum_temp += target_num
        i_target += 1

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)




class GenerativeDataset(data.Dataset):
    def __init__(self, args, dataset_name="style_GAN_init", datadir="./data",
            dataidxs=None,
            train=True, transform=None, target_transform=None,
            load_in_memory=False,
            image_resolution=32):

        self.args = args
        self.dataset_name = dataset_name

        if dataset_name in ["style_GAN_init", "style_GAN_init_64", "style_GAN_init_32_c62", "Gaussian_Noise",
                            "cifar_conv_decoder",
                            "style_GAN_init_32_c100","style_GAN_init_64_c200"]:
            self.labeled = True
        else:
            raise NotImplementedError
            # self.labeled = False
            # self.class_num = 0
            # self.classes = 1

        self.image_resolution = image_resolution

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.loader = default_loader
        if self.train:
            self.datadir = os.path.join(datadir, 'train')
        else:
            self.datadir = os.path.join(datadir, 'val')

        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()
        self.initial_local_data()


    def shuffle_data(self):
        # self.local_data = random.shuffle(self.local_data)
        random.shuffle(self.all_data)
        random.shuffle(self.local_data)


    def initial_local_data(self):
        if self.dataidxs == None:
            self.local_data = self.all_data
        elif type(self.dataidxs) == int:
            if self.alpha is not None:
                self.local_data = self.all_data[self.net_dataidx_map[self.dataidxs]]
            else:
                (begin, end) = self.net_dataidx_map[self.dataidxs]
                self.local_data = self.all_data[begin: end]
        else:
            # This is only suitable when not do dirichlet sampling
            assert self.alpha is None
            self.local_data = []
            for idxs in self.dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

        # self.data_num = sum(list(self.data_local_num_dict.values()))
        self.data_num = len(self.local_data)


    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(datadir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.datadir)
        self.classes = classes
        self.class_num = len(self.classes)
        self.class_to_idx = class_to_idx
        all_data, data_local_num_dict, net_dataidx_map = make_dataset(
            self.datadir, class_to_idx, IMG_EXTENSIONS,
            num_classes=1000, labeled=self.labeled)
        if len(all_data) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.datadir + "\n"
                "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        return all_data, data_local_num_dict, net_dataidx_map


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """

        path, target = self.local_data[index]
        img = self.loader(path)
        # logging.info(f"Before transform generative img.size: {img.size}")
        if self.transform is not None:
            img = self.transform(img)
        # logging.info(f"generative img.shape: {img.shape}")

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.local_data)












