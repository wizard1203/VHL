import logging
import copy

from utils.tracker import RuntimeTracker
from utils.metrics import Metrics
from utils.data_utils import (
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations
)
from utils.context import (
    raise_error_without_process,
    get_lock,
)

from data_preprocessing.loader import Data_Loader


class PSTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                train_data_num, device, args, model_trainer, perf_timer, metrics):
        self.args = args
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        if self.args.data_save_memory_mode:
            self.train_local_iter = None
        else:
            self.train_local_iter = iter(self.train_local)
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        logging.info(f"Initializing client: {self.client_index}"+
                    f" len(train_local) (local default iterations): {len(self.train_local)} local_sample_number: {self.local_sample_number}")
        # logging.info("self.local_sample_number: {}".format(self.local_sample_number))
        # assert len(self.train_local) == self.local_sample_number // args.batch_size
        self.test_local = self.test_data_local_dict[client_index]

        self.device = device
        self.local_num_iterations, self.global_num_iterations = \
            self.get_num_iterations()
        self.trainer = model_trainer
        # =============================================

    def get_num_iterations(self):
        local_num_iterations = get_local_num_iterations(self.local_sample_number, self.args.batch_size)
        global_num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
        return local_num_iterations, global_num_iterations


    def epoch_init(self):
        # if self.args.model in ['lstm', 'lstmwt2']:
        #     self.trainer.init_hidden()
        self.trainer.epoch_init()

    def epoch_end(self):
        self.trainer.epoch_end()


    def update_state(self, **kwargs):
        self.trainer.update_state(**kwargs)



    def lr_schedule(self, progress):
        self.trainer.lr_schedule(progress)

    def warmup_lr_schedule(self, iterations):
        self.trainer.warmup_lr_schedule(iterations)


    def set_model_params(self, weights):
        self.trainer.set_model_params(weights)

    def set_grad_params(self, named_grads):
        self.trainer.set_grad_params(named_grads)

    def clear_grad_params(self):
        self.trainer.clear_grad_params()

    def update_model_with_grad(self):
        self.trainer.update_model_with_grad()

    def clear_buffer(self):
        # if self.args.momentum > 0:
            # self.trainer.clear_momentum()
        if self.args.clear_buffer:
            self.trainer.clear_optim_buffer()

    def get_train_batch_data(self):
        try:
            train_batch_data = self.train_local_iter.next()
            logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(self.train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data


    def update_dataset(self, client_index, epoch):
        """
            No need to change dataset in real worker process, i.e. args.instantiate_all = True
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

        dataset_num_class = Data_Loader.num_classes_dict[self.args.dataset]



    def get_model_params(self):
        weights = self.trainer.get_model_params()
        if self.args.compression is None or self.args.compression == 'no':
            compressed_weights = weights
            model_indexes = None

        elif self.args.compression in ['topk','randomk', 'gtopk', 'randomkec', 'eftopk', 'gtopkef']:
            compressed_weights = {}
            model_indexes = {}
            for key in list(weights.keys()):
                logging.debug("weights[key].shape: {}, weights[key].numel(): {}".format(
                    weights[key].shape, weights[key].numel()
                ))
                _, model_indexes[key], compressed_weights[key] = \
                    self.compressor.compress(
                        self.compressor.flatten(weights[key]), name=key,
                        sigma_scale=3, ratio=self.args.compress_ratio
                    )
        elif self.args.compression in ['quantize', 'qsgd', 'sign']:
            compressed_weights = {}
            model_indexes = None
            for key in list(weights.keys()):
                logging.debug("weights[key].shape: {}, weights[key].numel(): {}".format(
                    weights[key].shape, weights[key].numel()
                ))
                compressed_weights[key] = self.compressor.compress(
                        weights[key], name=key,
                        quantize_level=self.args.quantize_level, is_biased=self.args.is_biased
                    )
        else:
            raise NotImplementedError

        return compressed_weights, model_indexes


    def get_model_diff_params(self, previous_model):
        weights = self.trainer.get_model_params()
        weights_diff = get_name_params_difference(previous_model, weights)

        if self.args.compression is None or self.args.compression == 'no':
            compressed_weights_diff = weights_diff
            model_indexes = None
        elif self.args.compression in ['topk','randomk', 'gtopk', 'randomkec', 'eftopk', 'gtopkef']:
            # weights_diff = get_name_params_difference(previous_model, weights)
            compressed_weights_diff = {}
            model_indexes = {}
            for key in list(weights_diff.keys()):
                logging.debug("weights_diff[key].shape: {}, weights_diff[key].numel(): {}".format(
                    weights_diff[key].shape, weights_diff[key].numel()
                ))
                _, model_indexes[key], compressed_weights_diff[key] = \
                    self.compressor.compress(
                        self.compressor.flatten(weights_diff[key]), name=key,
                        sigma_scale=3, ratio=self.args.compress_ratio
                    )
        elif self.args.compression in ['quantize', 'qsgd', 'sign']:
            compressed_weights_diff = {}
            model_indexes = None
            for key in list(weights_diff.keys()):
                logging.debug("weights_diff[key].shape: {}, weights_diff[key].numel(): {}".format(
                    weights_diff[key].shape, weights_diff[key].numel()
                ))
                compressed_weights_diff[key] = self.compressor.compress(
                        weights_diff[key], name=key,
                        quantize_level=self.args.quantize_level, is_biased=self.args.is_biased
                    )
        else:
            raise NotImplementedError

        return compressed_weights_diff, model_indexes


    def get_model_grads(self):
        named_grads = self.trainer.get_model_grads()
        # logging.debug(named_grads)
        if self.args.compression is None or self.args.compression == 'no':
            compressed_grads = named_grads
            grad_indexes = None
        elif self.args.compression in ['topk','randomk', 'gtopk', 'randomkec', 'eftopk', 'gtopkef']:
            compressed_grads = {}
            grad_indexes = {}
            for key in list(named_grads.keys()):
                logging.debug("named_grads[key].shape: {}, named_grads[key].numel(): {}".format(
                    named_grads[key].shape, named_grads[key].numel()
                ))
                _, grad_indexes[key], compressed_grads[key] = \
                    self.compressor.compress(
                        self.compressor.flatten(named_grads[key]), name=key,
                        sigma_scale=3, ratio=self.args.compress_ratio
                    )
        elif self.args.compression in ['quantize', 'qsgd', 'sign']:
            compressed_grads = {}
            grad_indexes = None
            for key in list(named_grads.keys()):
                logging.debug("named_grads[key].shape: {}, named_grads[key].numel(): {}".format(
                    named_grads[key].shape, named_grads[key].numel()
                ))
                compressed_grads[key] = self.compressor.compress(
                        named_grads[key], name=key,
                        quantize_level=self.args.quantize_level, is_biased=self.args.is_biased
                    )
        else:
            raise NotImplementedError

        return compressed_grads, grad_indexes


    def train_one_step(self, global_other_params, epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None
        ):
        if self.args.if_get_diff:
            previous_model = copy.deepcopy(self.trainer.get_model_params())
        # train_batch_data = next(self.train_local_iter)
        train_batch_data = self.get_train_batch_data()
        loss, pred, target = self.trainer.train_one_step(
            train_batch_data, device=self.device, args=self.args,
            epoch=epoch, iteration=iteration, end_of_epoch=end_of_epoch,
            tracker=tracker, metrics=metrics)

        if self.args.if_get_diff:
            compressed_weights_diff, model_indexes = self.get_model_diff_params(previous_model)
        else:
            compressed_weights_diff, model_indexes = self.get_model_params()

        return compressed_weights_diff, model_indexes, self.local_sample_number 


    def infer_bw_one_step(self, global_other_params, epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None
        ):
        # train_batch_data = next(self.train_local_iter)
        train_batch_data = self.get_train_batch_data()
        loss, pred, target = self.trainer.infer_bw_one_step(
            train_batch_data, device=self.device, args=self.args, 
            epoch=epoch, iteration=iteration, end_of_epoch=end_of_epoch,
            tracker=tracker, metrics=metrics)

        compressed_grads, grad_indexes = self.get_model_grads()
        # logging.debug(compressed_grads)
        return compressed_grads, grad_indexes, self.local_sample_number


    def local_test(self, epoch, tracker=None, metrics=None):
        self.trainer.test(self.test_local, self.device, self.args,
            epoch, tracker, metrics)

