import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import MNIST
import torch
import torchvision.transforms as transforms

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')



def data_transforms_mnist(resize=28, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):

    MNIST_MEAN = (0.1307,)
    MNIST_STD = (0.3081,)

    train_transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    image_size = 28

    if dataset_type == "full_dataset":
        pass
    elif dataset_type == "sub_dataset":
        train_transform.transforms.append(transforms.ToPILImage())
    else:
        raise NotImplementedError

    if resize is 28:
        pass
    else:
        image_size = resize
        train_transform.transforms.append(transforms.Resize(resize))
        test_transform.transforms.append(transforms.Resize(resize))

    if augmentation == "default":
        pass
    elif augmentation == "no":
        pass
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))

    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))


    return MNIST_MEAN, MNIST_STD, train_transform, test_transform



class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = mnist_dataobj.train_data
            data = mnist_dataobj.data
            target = mnist_dataobj.targets
            # target = np.array(mnist_dataobj.targets)
        else:
            data = mnist_dataobj.data
            target = mnist_dataobj.targets
            # target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)







class MNIST_truncated_WO_reload(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                full_dataset=None):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = mnist_dataobj.train_data
            data = self.full_dataset.data
            target = self.full_dataset.targets
            # target = np.array(mnist_dataobj.targets)
        else:
            data = self.full_dataset.data
            target = self.full_dataset.targets
            # target = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)










