import os
import logging

# from .FederatedEMNIST.data_loader import load_partition_data_federated_emnist
# from .fed_cifar100.data_loader import load_partition_data_federated_cifar100
# from .fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
# from .shakespeare.data_loader import load_partition_data_shakespeare
# from .shakespeare.iid_data_loader import load_iid_shakespeare
# from .stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
# from .MNIST.data_loader import load_partition_data_mnist
# from .ImageNet.data_loader import distributed_centralized_ImageNet_loader
# from .ImageNet.data_loader import load_partition_data_ImageNet
# from .TinyImageNet.data_loader import distributed_centralized_TinyImageNet_loader
# from .TinyImageNet.data_loader import load_partition_data_TinyImageNet
# from .TinyImageNet.data_loader import load_centralized_Tiny_ImageNet_200

# from ImageNet.data_loader import load_partition_data_ImageNet
# from Landmarks.data_loader import load_partition_data_landmarks

# from .MNIST.iid_data_loader import load_iid_mnist
# from .MNIST.centralized_loader import load_centralized_mnist
# from .cifar10.iid_data_loader import load_iid_cifar10
# from .cifar10.data_loader import load_partition_data_cifar10
# from .cifar10.centralized_loader import load_centralized_cifar10
# from .cifar100.data_loader import load_partition_data_cifar100
# from .cifar100.centralized_loader import load_centralized_cifar100
# from .cinic10.data_loader import load_partition_data_cinic10
# from .SVHN.data_loader import load_partition_data_SVHN
# from .SVHN.centralized_loader import load_centralized_SVHN
# from .ptb.iid_data_loader import load_iid_ptb
# from .FashionMNIST.iid_data_loader import load_iid_FashionMNIST
# from .FashionMNIST.data_loader import load_partition_data_fmnist

# from .generative.data_loader import get_dataloader_Generative



from .loader import Data_Loader
from .loader_shakespeare import Shakespeare_Data_Loader
from .generative_loader import Generative_Data_Loader

from .loader import NORMAL_DATASET_LIST
from .loader_shakespeare import SHAKESPEARE_DATASET_LIST
from .generative_loader import GENERATIVE_DATASET_LIST
from .FederatedEMNIST.data_loader import load_partition_data_federated_emnist



def get_new_datadir(args, datadir, dataset):
    # if "style_GAN_init" in dataset or "Gaussian" in dataset or "decoder" in dataset:
    if dataset in GENERATIVE_DATASET_LIST:
        return os.path.join(args.generative_dataset_root_path, dataset)
    else:
        return datadir



def load_data(load_as, args=None, process_id=0, mode="centralized", task="centralized", data_efficient_load=True,
                dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="iid", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default"):

    datadir = get_new_datadir(args, datadir, dataset)
    other_params = {}

    if task == "centralized":
        assert mode == "centralized"
        assert task == "centralized"
        if load_as == "training":
            if dataset in NORMAL_DATASET_LIST:
                data_loader = Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds
            elif dataset in SHAKESPEARE_DATASET_LIST:
                data_loader = Shakespeare_Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds

            elif dataset in GENERATIVE_DATASET_LIST:
                data_loader = Generative_Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds
            elif dataset == "femnist":
                client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
                data_local_num_dict, train_data_local_dict, test_data_local_dict, \
                class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir, args.batch_size)
                train_dl = train_data_global
                test_dl = test_data_global
            else:
                raise NotImplementedError
        elif load_as == "VHL_aux":
            if dataset in NORMAL_DATASET_LIST:
                data_loader = Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds
            elif dataset in GENERATIVE_DATASET_LIST:
                data_loader = Generative_Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
                    dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
                    data_sampler=data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
                train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_data()
                train_ds = data_loader.train_ds
                test_ds = data_loader.test_ds
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        other_params["train_ds"] = train_ds
        other_params["test_ds"] = test_ds
        return train_dl, test_dl, train_data_num, test_data_num, class_num, other_params
    else:
        if load_as == "training":
            if dataset in NORMAL_DATASET_LIST:
                data_loader = Data_Loader(args, process_id, mode, task, data_efficient_load, dirichlet_balance, dirichlet_min_p,
                    dataset, datadir, partition_method, partition_alpha, client_number, batch_size, num_workers,
                    data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                train_data_num, test_data_num, train_data_global, test_data_global, \
                    data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params \
                        = data_loader.load_data()
            elif dataset in SHAKESPEARE_DATASET_LIST:
                data_loader = Shakespeare_Data_Loader(args, process_id, mode, task, data_efficient_load, dirichlet_balance, dirichlet_min_p,
                    dataset, datadir, partition_method, partition_alpha, client_number, batch_size, num_workers,
                    data_sampler,
                    resize=resize, augmentation=augmentation, other_params=other_params)
                train_data_num, test_data_num, train_data_global, test_data_global, \
                    data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params \
                        = data_loader.load_data()
            elif dataset == "femnist":
                client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
                data_local_num_dict, train_data_local_dict, test_data_local_dict, \
                class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir, args.batch_size)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return train_data_num, test_data_num, train_data_global, test_data_global, \
                data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params







# def load_split_data(load_as, args=None, process_id=0, mode="centralized", task="centralized", data_efficient_load=True,
#                 dirichlet_balance=False, dirichlet_min_p=None,
#                 dataset="", datadir="./", partition_method="iid", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=1,
#                 data_sampler=None,
#                 resize=32, augmentation="default"):

#     datadir = get_new_datadir(args, datadir, dataset)
#     other_params = {}
#     traindata_cls_counts = {}
#     if load_as == "training":
#         data_loader = Data_Loader(args, process_id, mode, task, data_efficient_load, dirichlet_balance, dirichlet_min_p,
#             dataset, datadir, partition_method, partition_alpha, client_number, batch_size, num_workers,
#             data_sampler,
#             resize=resize, augmentation=augmentation, other_params=other_params)
#         train_data_num, test_data_num, train_data_global, test_data_global, \
#             data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts, other_params \
#                 = data_loader.load_split_data()
#     else:
#         raise NotImplementedError

#     other_params["traindata_cls_counts"] = traindata_cls_counts
#     return train_data_num, test_data_num, train_data_global, test_data_global, \
#             data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params



# def load_centralized_data(load_as, args, process_id, mode, task,
#             dataset, datadir, batch_size, num_workers,
#             data_sampler=None,
#             resize=32, augmentation="default"):

#     datadir = get_new_datadir(args, datadir, dataset)
#     other_params = {}
#     assert mode == "centralized"
#     assert task == "centralized"
#     if load_as == "training":
#         data_loader = Data_Loader(args=args, process_id=process_id, mode=mode, task=task,
#             dataset=dataset, datadir=datadir, batch_size=batch_size, num_workers=num_workers,
#             data_sampler=data_sampler,
#             resize=resize, augmentation=augmentation, other_params=other_params)
#         train_dl, test_dl, train_data_num, test_data_num, class_num, other_params = data_loader.load_centralized_data()
#         train_ds = data_loader.train_ds
#         test_ds = data_loader.test_ds
#     elif load_as == "VHL_aux":
#         pass
#     else:
#         raise NotImplementedError
#     other_params["train_ds"] = train_ds
#     other_params["test_ds"] = test_ds

#     return train_dl, test_dl, train_data_num, test_data_num, class_num, other_params




# def load_data(args, dataset_name, **kwargs):
#     other_params = {}
#     traindata_cls_counts = {}
#     if dataset_name == "mnist":
#         if args.partition_method == 'iid':
#             train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = load_iid_mnist(args.dataset, args.data_dir, args.partition_method,
#                     args.partition_alpha, args.client_num_in_total, args.batch_size, args.client_index)
#         else:
#             logging.info("load_data. dataset_name = %s" % dataset_name)
#             # client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#             # train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             # class_num = load_partition_data_mnist(args.batch_size)
#             train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = load_partition_data_mnist(args.dataset, args.data_dir, args.partition_method,
#                                     args.partition_alpha, args.client_num_in_total, args.batch_size, args)
#             """
#             For shallow NN or linear models, 
#             we uniformly sample a fraction of clients each round (as the original FedAvg paper)
#             """
#             # args.client_num_in_total = client_num

#     elif dataset_name == "femnist":
#         logging.info("load_data. dataset_name = %s" % dataset_name)
#         client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
#         args.client_num_in_total = client_num
#     elif dataset_name == "shakespeare":
#         if args.partition_method == 'iid':
#             logging.info("load_data. dataset_name = %s" % dataset_name)
#             client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = load_iid_shakespeare(args.data_dir,
#                                             args.batch_size, args.client_index, args)
#             args.client_num_in_total = client_num
#         else:
#             logging.info("load_data. dataset_name = %s" % dataset_name)
#             client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = load_partition_data_shakespeare(args.batch_size)
#             args.client_num_in_total = client_num
#     elif dataset_name == "fed_shakespeare":
#         logging.info("load_data. dataset_name = %s" % dataset_name)
#         client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)
#         args.client_num_in_total = client_num

#     elif dataset_name == "fed_cifar100":
#         # assert args.num_classes == 100
#         logging.info("load_data. dataset_name = %s" % dataset_name)
#         client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)
#         args.client_num_in_total = client_num
#     elif dataset_name == "stackoverflow_lr":
#         logging.info("load_data. dataset_name = %s" % dataset_name)
#         client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)
#         args.client_num_in_total = client_num
#     elif dataset_name == "stackoverflow_nwp":
#         logging.info("load_data. dataset_name = %s" % dataset_name)
#         client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
#         args.client_num_in_total = client_num
#     # loading ImageNet should be updated
#     # elif dataset_name == "ILSVRC2012":
#     #     if args.partition_method == 'iid':
#     #         train_data_num, test_data_num, train_data_global, test_data_global, \
#     #         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#     #         class_num = distributed_centralized_ImageNet_loader(dataset=dataset_name, data_dir=args.data_dir,
#     #             world_size=args.client_num_in_total, rank=args.client_index,
#     #             batch_size=args.batch_size, args=args)
#     #     elif args.partition_method == 'hetero':
#     #         logging.info("load_data. dataset_name = %s" % dataset_name)
#     #         train_data_num, test_data_num, train_data_global, test_data_global, \
#     #         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#     #         class_num = load_partition_data_ImageNet(dataset=dataset_name, data_dir=args.data_dir,
#     #             partition_method=None, partition_alpha=None, 
#     #             client_number=args.client_num_in_total, batch_size=args.batch_size)
#     elif dataset_name == "Tiny-ImageNet-200":
#         # assert args.num_classes == 200
#         if args.partition_method == 'iid':
#             train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = distributed_centralized_TinyImageNet_loader(dataset=dataset_name, data_dir=args.data_dir,
#                 client_number=args.client_num_in_total, rank=args.client_index,
#                 batch_size=args.batch_size, args=args)
#         elif args.partition_method == 'hetero':
#             logging.info("load_data. dataset_name = %s" % dataset_name)
#             train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = load_partition_data_TinyImageNet(dataset=dataset_name, data_dir=args.data_dir,
#                 partition_method=args.partition_method, partition_alpha=args.partition_alpha, 
#                 client_number=args.client_num_in_total, batch_size=args.batch_size, args=args)

#     elif dataset_name == "gld23k":
#         logging.info("load_data. dataset_name = %s" % dataset_name)
#         args.client_num_in_total = 233
#         fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
#         fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')
#         args.data_dir = os.path.join(args.data_dir, 'images')

#         train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
#             fed_train_map_file=fed_train_map_file, fed_test_map_file=fed_test_map_file,
#             partition_method=None, partition_alpha=None, 
#             client_number=args.client_num_in_total, batch_size=args.batch_size)

#     elif dataset_name == "gld160k":
#         logging.info("load_data. dataset_name = %s" % dataset_name)
#         args.client_num_in_total = 1262
#         fed_train_map_file = os.path.join(args.data_dir, 'federated_train.csv')
#         fed_test_map_file = os.path.join(args.data_dir, 'test.csv')
#         args.data_dir = os.path.join(args.data_dir, 'images')

#         train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_partition_data_landmarks(dataset=dataset_name, data_dir=args.data_dir,
#             fed_train_map_file=fed_train_map_file, fed_test_map_file=fed_test_map_file,
#             partition_method=None, partition_alpha=None, 
#             client_number=args.client_num_in_total, batch_size=args.batch_size)
#     elif dataset_name == "cifar10" and args.partition_method == 'iid':
#         train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = load_iid_cifar10(args.dataset, args.data_dir, args.partition_method,
#                 args.partition_alpha, args.client_num_in_total, args.batch_size, args.client_index, args)
#     elif dataset_name == "fmnist":
#         if args.partition_method == 'iid':
#             train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = load_iid_FashionMNIST(args.dataset, args.data_dir, args.partition_method,
#                     args.partition_alpha, args.client_num_in_total, args.batch_size, args.client_index, args)
#         else:
#             train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num = load_partition_data_fmnist(args.dataset, args.data_dir, args.partition_method,
#                                     args.partition_alpha, args.client_num_in_total, args.batch_size, args)
#     elif dataset_name == "cifar100" and args.partition_method == 'iid':
#         train_data_local_num_dict = None
#         train_data_local_dict = None
#         test_data_local_dict = None
#         train_data_global, test_data_global, train_data_num, test_data_num, class_num \
#             = load_centralized_cifar100(args.dataset, args.data_dir, args.batch_size, args=args)

#     elif dataset_name == "ptb":
#         if args.partition_method == 'iid':
#             train_data_num, test_data_num, train_data_global, test_data_global, \
#             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#             class_num, other_params = load_iid_ptb(args.dataset, args.data_dir, args.partition_method,
#                     args.partition_alpha, args.client_num_in_total, args.batch_size,
#                     args.lstm_num_steps, args.client_index)
#             logging.info("vocab_size: {}, batch_size :{}, num_steps:{} ".format(
#                 other_params["vocab_size"], args.batch_size, args.lstm_num_steps))
#         else:
#             raise NotImplementedError

#     else:
#         if dataset_name == "cifar10":
#             data_loader = load_partition_data_cifar10
#         elif dataset_name == "cifar100":
#             # assert args.num_classes == 100
#             data_loader = load_partition_data_cifar100
#         elif dataset_name == "cinic10":
#             data_loader = load_partition_data_cinic10
#         elif dataset_name == "SVHN":
#             data_loader = load_partition_data_SVHN
#         else:
#             data_loader = load_partition_data_cifar10

#         # train_data_num, test_data_num, train_data_global, test_data_global, \
#         # train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         # class_num, traindata_cls_counts = data_loader(args.dataset, args.data_dir, args.partition_method,
#         #                         args.partition_alpha, args.client_num_in_total, args.batch_size, args)
#         train_data_num, test_data_num, train_data_global, test_data_global, \
#         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
#         class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
#                                 args.partition_alpha, args.client_num_in_total, args.batch_size, args)

#     other_params["traindata_cls_counts"] = traindata_cls_counts
#     dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
#                train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params]
#     return dataset




# def load_centralized_data(args, dataset_name, **kwargs):
#     other_params = {}

#     # assert "usage" in kwargs

#     if dataset_name == "mnist":
#         # if "usage" in kwargs and kwargs["usage"] == "task":
#         if kwargs.get("usage", "task") == "task":
#             # assert args.num_classes == 10
#             batch_size = args.batch_size
#             data_dir = args.data_dir
#         train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_mnist(
#             dataset_name, data_dir=data_dir, batch_size=batch_size, args=args)
#     elif dataset_name == "cifar10":
#         # if kwargs["usage"] == "task":
#         if kwargs.get("usage", "task") == "task":
#             # assert args.num_classes == 10
#             batch_size = args.batch_size
#             data_dir = args.data_dir
#         train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_cifar10(
#             dataset_name, data_dir=data_dir, batch_size=batch_size, args=args)
#     elif dataset_name == "cifar100":
#         if kwargs.get("usage", "task") == "task":
#             # assert args.num_classes == 100
#             batch_size = args.batch_size
#             data_dir = args.data_dir
#         train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_cifar100(
#             dataset_name, data_dir=data_dir, batch_size=batch_size, args=args)
#     # elif dataset_name == "style_GAN_init":
#     elif dataset_name == "Tiny-ImageNet-200":
#         if kwargs.get("usage", "task") == "task":
#             # assert args.num_classes == 200
#             batch_size = args.batch_size
#             data_dir = args.data_dir
#         train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_cifar100(
#             dataset_name, data_dir=data_dir, batch_size=batch_size, args=args)
#     elif dataset_name == "Sub-Tiny-ImageNet-200":
#         assert "max_classes" in kwargs
#         if kwargs.get("usage", "task") == "task":
#             # assert args.num_classes == 200
#             batch_size = args.batch_size
#             data_dir = args.data_dir
#             image_size = 64
#         else:
#             batch_size = kwargs["batch_size"]
#             image_size = kwargs["image_size"]
#             data_dir = kwargs["data_dir"]

#         train_dl, test_dl, train_data_num, test_data_num, class_num = load_centralized_Tiny_ImageNet_200(
#             dataset_name, data_dir=data_dir, batch_size=batch_size, args=args, max_classes=kwargs["max_classes"],
#             image_size=image_size)
#         other_params["train_ds"] = train_dl.dataset
#         other_params["test_ds"] = test_dl.dataset
#         logging.info(f"Loading Sub-Tiny-ImageNet-200,  train_data_num: {train_data_num}, test_data_num: {test_data_num}\
#             class_num: {class_num}")
#     elif "style_GAN_init" in dataset_name or "Gaussian" in dataset_name or "decoder" in dataset_name:
#         datadir = os.path.join(args.generative_dataset_root_path, dataset_name)

#         resize_size = args.generative_dataset_resize

#         train_dl, test_dl, train_ds, test_ds = get_dataloader_Generative(
#                         datadir, train_bs=kwargs["batch_size"], test_bs=kwargs["batch_size"],
#                         dataidxs=None, args=args,
#                         full_train_dataset=None, full_test_dataset=None,
#                         dataset_name=dataset_name, 
#                         resize_size=resize_size, image_resolution=args.image_resolution,
#                         load_in_memory=args.generative_dataset_load_in_memory)

#         train_data_num = train_ds.data_num
#         if test_ds is not None:
#             test_data_num = test_ds.data_num
#         else:
#             test_data_num = None

#         class_num = train_ds.class_num
#         other_params["train_ds"] = train_ds
#         other_params["test_ds"] = test_ds
#     else:
#         raise NotImplementedError

#     return train_dl, test_dl, train_data_num, test_data_num, class_num, other_params




def load_multiple_centralized_dataset(load_as, args, process_id, mode, task,
                        dataset_list, datadir_list, batch_size, num_workers,
                        data_sampler=None,
                        resize=32, augmentation="default"): 
    train_dl_dict = {}
    test_dl_dict = {}
    train_ds_dict = {}
    test_ds_dict = {}
    class_num_dict = {}
    train_data_num_dict = {}
    test_data_num_dict = {}

    for i, dataset in enumerate(dataset_list):
        # kwargs["data_dir"] = datadir_list[i]
        datadir = datadir_list[i]
        # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
        #     = load_centralized_data(load_as, args, process_id, mode, task,
        #                 dataset, datadir, batch_size, num_workers,
        #                 data_sampler=None,
        #                 resize=resize, augmentation=augmentation)
        train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
            = load_data(load_as=load_as, args=args, process_id=process_id,
                        mode="centralized", task="centralized",
                        dataset=dataset, datadir=datadir, batch_size=args.batch_size, num_workers=args.data_load_num_workers,
                        data_sampler=None,
                        resize=resize, augmentation=augmentation)

        train_dl_dict[dataset] = train_dl
        test_dl_dict[dataset] = test_dl
        train_ds_dict[dataset] = other_params["train_ds"]
        test_ds_dict[dataset] = other_params["test_ds"]
        class_num_dict[dataset] = class_num
        train_data_num_dict[dataset] = train_data_num
        test_data_num_dict[dataset] = test_data_num

    return train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict, \
        class_num_dict, train_data_num_dict, test_data_num_dict


def VHL_load_dataset(args, process_id=0):
    kwargs = {}
    # kwargs["batch_size"] = args.VHL_dataset_batch_size // len(args.VHL_dataset_list)
    # kwargs["usage"] = "auxiliary"
    # kwargs["image_size"] = args.dataset_load_image_size
    batch_size = args.VHL_dataset_batch_size // len(args.VHL_dataset_list)
    image_size = args.dataset_load_image_size

    logging.info(f"Loading datasets for fednoise ------  VHL_dataset_list :{args.VHL_dataset_list}")

    dataset_list = []
    datadir_list = []

    # May we support multiple dataset in the future.
    for i, dataset_name in enumerate(args.VHL_dataset_list):
        if dataset_name == "Sub-Tiny-ImageNet-200":
            # assert not args.num_classes > 200
            kwargs["max_classes"] = args.num_classes
            dataset_list.append(dataset_name)
            datadir_list.append(args.VHL_datadir_list[i])

        elif "style_GAN_init" in dataset_name or "Gaussian" in dataset_name or "decoder" in dataset_name:
            dataset_list.append(dataset_name)
            datadir = os.path.join(args.generative_dataset_root_path, dataset_name)
            datadir_list.append(datadir)
        else:
            raise NotImplementedError

    if args.VHL_data == "dataset":
        # if in args.VHL_dataset_list:
        train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict, \
            class_num_dict, train_data_num_dict, test_data_num_dict = load_multiple_centralized_dataset(
                load_as="VHL_aux", args=args, process_id=process_id, mode="centralized", task="centralized",
                dataset_list=dataset_list, datadir_list=datadir_list, batch_size=batch_size, num_workers=1,
                data_sampler=None,
                resize=image_size, augmentation="default")
    else:
        raise NotImplementedError

    return train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict











