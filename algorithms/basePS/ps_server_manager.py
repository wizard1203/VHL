import logging
import os
import sys
from abc import ABC, abstractmethod

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager

from utils.tracker import RuntimeTracker

from .message_define import MyMessage

from timers.server_timer import ServerTimer


class PSServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", perf_timer=None, metrics=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.global_num_iterations = self.aggregator.global_num_iterations
        self.max_comm_round = self.get_max_comm_round()
        # assert args.client_num_in_total == self.size - 1
        # assert args.client_num_per_round == self.size - 1
        self.selected_clients = None
        # ================================================
        self.metrics = metrics
        self.server_timer = ServerTimer(
            self.args,
            self.global_num_iterations,
            local_num_iterations_dict=None
        )
        self.total_train_tracker = RuntimeTracker(
            mode='Train',
            things_to_metric=self.metrics.metric_names,
            timer=self.server_timer,
            args=args
        )
        self.total_test_tracker = RuntimeTracker(
            mode='Test',
            things_to_metric=self.metrics.metric_names,
            timer=self.server_timer,
            args=args
        )

    def run(self):
        super().run()

    @abstractmethod
    def get_max_comm_round(self):
        """
            Maybe this is only useful in FedAvg.
        """
        pass

    def epoch_init(self):
        self.aggregator.epoch_init()

    def check_end_epoch(self):
        return (self.server_timer.global_outer_iter_idx > 0 and self.server_timer.global_outer_iter_idx % self.global_num_iterations == 0)

    def check_test_frequency(self):
        return self.server_timer.global_outer_epoch_idx % self.args.frequency_of_the_test == 0 or self.server_timer.global_outer_epoch_idx == self.args.max_epochs - 1

    def check_end_training(self):
        if self.server_timer.global_comm_round_idx == self.max_comm_round + 1:
            self.total_train_tracker.upload_record_to_wandb()
            self.total_test_tracker.upload_record_to_wandb()
            self.finish()
            return

    def optimize_with_grad(self, grad_params):
        self.aggregator.set_grad_params(grad_params)
        self.aggregator.update_model_with_grad()


    def test(self):
        logging.info("################test_on_server_for_all_clients : {}".format(
            self.server_timer.global_outer_epoch_idx))
        self.aggregator.test_on_server_for_all_clients(
            self.server_timer.global_outer_epoch_idx, self.total_test_tracker, self.metrics)

        self.total_train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)
        self.total_test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=self.metrics)


    def check_and_test(self):
        if self.check_end_epoch():
            if self.check_test_frequency():
                self.test()
            else:
                self.total_train_tracker.reset()
                self.total_test_tracker.reset()


    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.server_timer.global_outer_epoch_idx
        iterations = self.server_timer.global_outer_iter_idx

        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.aggregator.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.aggregator.trainer.lr_schedule(epochs)


    def send_init_msg(self):
        # sampling clients
        logging.debug("send_init_msg")

        global_other_params = {}

        time_info=self.server_timer.get_time_info_to_send()
        logging.info(f"time_info: {time_info}")
        client_indexes = self.aggregator.client_sampling(
            self.server_timer.global_comm_round_idx, self.args.client_num_in_total,
            self.args.client_num_per_round)
        self.selected_clients = client_indexes
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in client_indexes:
            self.send_message_init_config(
                process_id+1, global_model_params, process_id,
                global_other_params=global_other_params,
                time_info=time_info)

    def get_tracker_info_from_message(self, client_index, msg_params):
        local_train_tracker_info = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_TRACKER_INFO)
        local_test_tracker_info = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_TRACKER_INFO)

        if local_train_tracker_info is not None:
            logging.debug('Server: receive train_tracker_info')
            # assert local_train_tracker_info['n_samples'] > 0
            self.total_train_tracker.decode_local_info(client_index, local_train_tracker_info)

        if local_test_tracker_info is not None:
            logging.debug('Server: receive test_tracker_info')
            # assert local_test_tracker_info['n_samples'] > 0
            self.total_test_tracker.decode_local_info(client_index, local_test_tracker_info)
        return local_train_tracker_info, local_test_tracker_info


    def handle_message_receive_model_from_client(self, msg_params):
        sender_id, model_params, model_indexes, local_sample_number, client_other_params, time_info, \
            local_train_tracker_info, local_test_tracker_info = self.process_sync_model_message(msg_params)

        client_index = sender_id - 1
        self.algorithm_on_handle_message_receive_model_from_client(
            sender_id, client_index, model_params, model_indexes, local_sample_number,
            client_other_params, time_info,
            local_train_tracker_info, local_test_tracker_info
        )


    @abstractmethod
    def algorithm_on_handle_message_receive_model_from_client(
        self, sender_id, client_index, model_params, model_indexes, local_sample_number, 
        client_other_params, time_info,
        local_train_tracker_info, local_test_tracker_info
    ):
        pass


    def handle_message_receive_grad_from_client(self, msg_params):
        sender_id, grad_params, grad_indexes, local_sample_number, client_other_params, time_info, \
            local_train_tracker_info, local_test_tracker_info = self.process_sync_grad_message(msg_params)

        client_index = sender_id - 1
        self.algorithm_on_handle_message_receive_grad_from_client(
            sender_id, client_index, grad_params, grad_indexes, local_sample_number, 
            client_other_params, time_info,
            local_train_tracker_info, local_test_tracker_info
        )


    @abstractmethod
    def algorithm_on_handle_message_receive_grad_from_client(
        self, sender_id, client_index, grad_params, grad_indexes, local_sample_number, 
        client_other_params, time_info,
        local_train_tracker_info, local_test_tracker_info
    ):
        pass



    def choose_clients_and_send(self, global_params, params_type='grad', global_other_params=None):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.server_timer.global_comm_round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round
        )
        logging.debug("size = %d" % self.size)

        time_info = self.server_timer.get_time_info_to_send()
        self.selected_clients = client_indexes
        for receiver_id in client_indexes:
            if params_type == 'grad':
                self.send_message_sync_grad_to_client(
                    receiver_id+1, global_params, receiver_id, 
                    global_other_params=global_other_params,
                    time_info=time_info)
            elif params_type == 'model':
                self.send_message_sync_model_to_client(
                    receiver_id+1, global_params, receiver_id,
                    global_other_params=global_other_params,
                    time_info=time_info)


    def send_message_init_config(self, receive_id, global_model_params, client_index, 
                                global_other_params=None,
                                time_info=None):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS, global_other_params)
        message.add_params(MyMessage.MSG_ARG_KEY_TIME_INFO, time_info)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index,
                                        global_other_params=None,
                                        time_info=None):
        logging.debug("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS, global_other_params)
        message.add_params(MyMessage.MSG_ARG_KEY_TIME_INFO, time_info)
        self.send_message(message)

    def send_message_sync_grad_to_client(self, receive_id, global_grad_params, client_index,
                                        global_other_params=None,
                                        time_info=None):
        logging.debug("send_message_sync_grad_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_GRAD_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRAD_PARAMS, global_grad_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_OTHER_PARAMS, global_other_params)
        message.add_params(MyMessage.MSG_ARG_KEY_TIME_INFO, time_info)
        self.send_message(message)

    def process_sync_grad_message(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        grad_params = msg_params.get(MyMessage.MSG_ARG_KEY_GRAD_PARAMS)
        grad_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_GRAD_INDEXES)
        client_other_params = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        time_info = msg_params.get(MyMessage.MSG_ARG_KEY_TIME_INFO)
        self.server_timer.update_time_info(time_info)

        client_index = sender_id - 1 
        local_train_tracker_info, local_test_tracker_info = \
            self.get_tracker_info_from_message(client_index, msg_params)

        return sender_id, grad_params, grad_indexes, local_sample_number, \
            client_other_params, time_info, \
            local_train_tracker_info, local_test_tracker_info

    def process_sync_model_message(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        model_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_INDEXES)
        client_other_params = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_OTHER_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        time_info = msg_params.get(MyMessage.MSG_ARG_KEY_TIME_INFO)
        self.server_timer.update_time_info(time_info)

        client_index = sender_id - 1
        local_train_tracker_info, local_test_tracker_info = \
            self.get_tracker_info_from_message(client_index, msg_params)

        return sender_id, model_params, model_indexes, local_sample_number, \
            client_other_params, time_info, \
            local_train_tracker_info, local_test_tracker_info





