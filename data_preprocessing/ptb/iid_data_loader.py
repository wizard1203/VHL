import os
import argparse
import time
import math
import logging

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler

from . import ptb_reader




def load_iid_ptb(dataset, data_dir, partition_method, 
        partition_alpha, client_number, batch_size, lstm_num_steps,
        rank=0):
    """
    partition_method and partition_alpha are not used.
    """

    other_params = {}
    # Data loading code

    # =====================================
    # lstm_num_steps = 35
    # hidden_size = 1500
    # =================================

    raw_data = ptb_reader.ptb_raw_data(data_path=data_dir, prefix='ptb')
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
    vocab_size = len(word_to_id)
    other_params["vocab_size"] = vocab_size

    input_shape = (batch_size, lstm_num_steps)
    output_shape = (batch_size, lstm_num_steps)

    logging.info('Vocabluary size: {}'.format(vocab_size))
    logging.info('load data')

    epoch_size = ((len(train_data) // batch_size) - 1) // lstm_num_steps

    train_dataset = ptb_reader.TrainDataset(train_data, batch_size, lstm_num_steps)
    test_dataset = ptb_reader.TestDataset(valid_data, batch_size, lstm_num_steps)
    #test_set = ptb_reader.TestDataset(test_data, self.batch_size, self.lstm_num_steps)

    train_sampler = None
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
                                shuffle=shuffle, num_workers=4, drop_last=True)
    test_data_global = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    train_sampler = train_sampler
    train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4, sampler=train_sampler, drop_last=True)
    test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_index in range(client_number):
        train_data_local_dict[client_index] = train_dl
        test_data_local_dict[client_index] = test_dl
        data_local_num_dict[client_index] = train_data_num // client_number
        logging.info("client_index = %d, local_sample_number = %d" % (client_index, train_data_num))


    logging.info('=========****** finish getting ptb data, num_of_training samples: %d ===========' % \
                len(train_data))

    class_num = None
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, other_params






