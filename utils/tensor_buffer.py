import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch

from .data_utils import (
    flatten,
    get_data,
)


"""
    Refer to ChocoSGD code
"""

class TensorBuffer(object):
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors, use_cuda=True):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]

        self.buffer = flatten(tensors, use_cuda=use_cuda)  # copies

    def __getitem__(self, index):
        # logging.debug("self.buffer: %s" % self.buffer)
        # logging.debug("self._start_idx: %s" % self._start_idx)
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def unpack(self, tensors):
        assert len(tensors) == len(self)
        for tensor, entry in zip(tensors, self):
            tensor.data[:] = entry



def recover_params(
    param_groups, param_names, rank=None, neighbor_hat_params=None, get_hat_params=True
):
    # get flattened params.
    params, _ = get_data(param_groups, param_names, is_get_grad=False)
    flatten_params = TensorBuffer(params)

    if get_hat_params:
        assert neighbor_hat_params is not None and rank is not None
        # recover the hat_params.
        flatten_hat_params = TensorBuffer(params)
        flatten_hat_params.buffer.data[:] = neighbor_hat_params[rank].buffer
        return params, flatten_params, flatten_hat_params
    else:
        return params, flatten_params


def update_params_from_neighbor(
    neighbor_hat_params, flatten_params, consensus_stepsize, self_rank
):
    flatten_params.buffer += consensus_stepsize * (
        neighbor_hat_params["memory"].buffer - neighbor_hat_params[self_rank].buffer
    )




