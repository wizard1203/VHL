from abc import ABC, abstractmethod

import logging

class Timer(ABC):
    def __init__(self, args):
        """
            num_iterations_dict contains number of iterations of each epoch of each client.
            This may vary across different clients.
        """
        self.args = args

        # global timestamp
        self.global_comm_round_idx = 0
        self.global_outer_epoch_idx = 0
        self.global_inner_epoch_idx = 0
        self.global_outer_iter_idx = 0
        self.global_inner_iter_idx = 0

        # local timestamp dict
        self.local_comm_round_idx_dict = {}
        self.local_outer_epoch_idx_dict = {}
        self.local_inner_epoch_idx_dict = {}
        self.local_outer_iter_idx_dict = {}
        self.local_inner_iter_idx_dict = {}


    def update_global_time_info(self, global_time_info):
        self.global_comm_round_idx = global_time_info["global_comm_round_idx"]
        self.global_outer_epoch_idx = global_time_info["global_outer_epoch_idx"]
        self.global_inner_epoch_idx = global_time_info["global_inner_epoch_idx"]
        self.global_outer_iter_idx = global_time_info["global_outer_iter_idx"]
        self.global_inner_iter_idx = global_time_info["global_inner_iter_idx"]


    def update_local_time_info_dict(self, local_time_info_dict):
        if local_time_info_dict["update_local_time_info_dict"] is True:
            logging.warning("WARNING!!! receive local_time_info_dict, which is usually updated by server, Be careful.")
            self.local_comm_round_idx_dict.update(local_time_info_dict["local_comm_round_idx_dict"])
            self.local_outer_epoch_idx_dict.update(local_time_info_dict["local_outer_epoch_idx_dict"])
            self.local_inner_epoch_idx_dict.update(local_time_info_dict["local_inner_epoch_idx_dict"])
            self.local_outer_iter_idx_dict.update(local_time_info_dict["local_outer_iter_idx_dict"])
            self.local_inner_iter_idx_dict.update(local_time_info_dict["local_inner_iter_idx_dict"])

    def get_global_time_info(self):
        global_time_info = {}
        global_time_info["global_comm_round_idx"] = self.global_comm_round_idx
        global_time_info["global_outer_epoch_idx"] = self.global_outer_epoch_idx
        global_time_info["global_inner_epoch_idx"] = self.global_inner_epoch_idx
        global_time_info["global_outer_iter_idx"] = self.global_outer_iter_idx
        global_time_info["global_inner_iter_idx"] = self.global_inner_iter_idx
        return global_time_info


    def get_local_time_info_dict(self, update_local_time_info_dict=False):
        local_time_info_dict = {}
        local_time_info_dict["update_local_time_info_dict"] = update_local_time_info_dict
        local_time_info_dict["local_comm_round_idx_dict"] = self.local_comm_round_idx_dict
        local_time_info_dict["local_outer_epoch_idx_dict"] = self.local_outer_epoch_idx_dict
        local_time_info_dict["local_inner_epoch_idx_dict"] = self.local_inner_epoch_idx_dict
        local_time_info_dict["local_outer_iter_idx_dict"] = self.local_outer_iter_idx_dict
        local_time_info_dict["local_inner_iter_idx_dict"] = self.local_inner_iter_idx_dict
        return local_time_info_dict

    @abstractmethod
    def update_time_info(self, time_info):
        pass

    @abstractmethod
    def past_comm_round(self, comm_round=1):
        pass

    @abstractmethod
    def past_iterations(self, iterations=1):
        pass

    @abstractmethod
    def past_epochs(self, epochs):
        pass

















