import logging
import copy

from algorithms.basePS.ps_client_trainer import PSTrainer


from compression.compression import compressors
from utils.perf_timer import Perf_Timer
from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.data_utils import (
    get_name_params_difference
)
from utils.context import (
    raise_error_without_process,
    get_lock,
)


class FedAVGTrainer(PSTrainer):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer, metrics):
        super().__init__(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer, metrics)

    # Do not need to get local data iterator
    # @override
    def update_dataset(self, client_index, epoch):
        """
        """
        if self.args.instantiate_all:
            assert self.client_index == client_index
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

        with raise_error_without_process():
            logging.debug("type(self.train_local): {}".format(type(self.train_local)))
            logging.debug("type(self.train_local.sampler): {}".format(type(self.train_local.sampler)))
            # This is used for distributed sampler
            self.train_local.sampler.set_epoch(epoch)


    def fedavg_train(self, round_idx=None, global_other_params=None, tracker=None, metrics=None, if_get_diff=True):

        if self.args.if_get_diff:
            previous_model = copy.deepcopy(self.trainer.get_model_params())
        # In new version, we do not use these
        # if self.args.client_optimizer == 'FedProx':
        #     self.model_trainer.optimizer.update_old_init()
        # elif self.args.client_optimizer == 'FedNova':
        #     pass
        else:
            pass
        for epoch in range(self.args.global_epochs_per_round):
            self.epoch_init()
            self.trainer.train_one_epoch(self.train_local, self.device, self.args, epoch, tracker, metrics)

        if self.args.if_get_diff:
            compressed_weights_diff, model_indexes = self.get_model_diff_params(previous_model)
        else:
            compressed_weights_diff, model_indexes = self.get_model_params()

        return compressed_weights_diff, model_indexes, self.local_sample_number

    # Not support testing on client now.
    # @overrid
    def test(self):
        # for warning
        raise NotImplementedError

