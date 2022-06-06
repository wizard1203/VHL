from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import collections
import os
import numpy as np

def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, prefix="ptb"):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
        data_path: string path to the directory where simple-examples.tgz has
        been extracted.
    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

class TrainDataset(Dataset):
    def __init__(self, raw_data, batch_size, num_steps):
        self.raw_data = np.array(raw_data, dtype=np.int64)
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_len = len(self.raw_data)
        self.sample_len = self.data_len // self.num_steps

    def __getitem__(self, idx):
        
        num_steps_begin_index = self.num_steps * idx
    
        num_steps_end_index = self.num_steps * (idx + 1)
        
        # print("num_steps_end_index  :  %d== ",num_steps_end_index)
        x = self.raw_data[num_steps_begin_index : num_steps_end_index]
        y = self.raw_data[num_steps_begin_index + 1 : num_steps_end_index + 1]
        
        return (x, y)
    
    def __len__(self):
        return self.sample_len - self.sample_len % self.batch_size

class TestDataset(Dataset):
    def __init__(self, raw_data, batch_size, num_steps):
        self.raw_data = np.array(raw_data, dtype=np.int64)
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.data_len = len(self.raw_data)
        self.sample_len = self.data_len // self.num_steps
        # self.batch_len = self.sample_len // self.batch_size - 1
    
    def __getitem__(self, idx):
        num_steps_begin_index = self.num_steps * idx
        
        num_steps_end_index = self.num_steps * (idx + 1)
        
        # print("num_steps_end_index  :  %d== ",num_steps_end_index)
        x = self.raw_data[num_steps_begin_index: num_steps_end_index]
        y = self.raw_data[num_steps_begin_index + 1: num_steps_end_index + 1]
        
        return (x, y)
    
    def __len__(self):
        return self.sample_len - self.sample_len % self.batch_size


