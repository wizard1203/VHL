import logging

from .base_timer import Timer


class ClientTimer(Timer):
    def __init__(self, args, global_num_iterations, local_num_iterations_dict, client_index=None):
        super().__init__(args)
        self.role = 'client'
        if client_index is None:
            self.client_index = args.client_index
        else:
            self.client_index = client_index
        self.global_num_iterations = global_num_iterations
        self.local_num_iterations_dict = local_num_iterations_dict
        self.local_num_iterations = local_num_iterations_dict[self.client_index]

        # self local timestamp
        self.local_comm_round_idx = 0
        self.local_outer_epoch_idx = 0
        self.local_inner_epoch_idx = 0
        self.local_outer_iter_idx = 0
        self.local_inner_iter_idx = 0


    def update_local_time_info(self, local_time_info):
        if local_time_info["client_index"] == self.client_index:
            logging.warning("WARNING!!! Client {}, receive self time info, Be careful.".format(
                self.client_index
            ))
            self.local_comm_round_idx = local_time_info["local_comm_round_idx"]
            self.local_outer_epoch_idx = local_time_info["local_outer_epoch_idx"]
            self.local_inner_epoch_idx = local_time_info["local_inner_epoch_idx"]
            self.local_outer_iter_idx = local_time_info["local_outer_iter_idx"]
            self.local_inner_iter_idx = local_time_info["local_inner_iter_idx"]
            self.local_comm_round_idx_dict[self.client_index] = self.local_comm_round_idx
            self.local_outer_epoch_idx_dict[self.client_index] = self.local_outer_epoch_idx
            self.local_inner_epoch_idx_dict[self.client_index] = self.local_inner_epoch_idx
            self.local_outer_iter_idx_dict[self.client_index] = self.local_outer_iter_idx
            self.local_inner_iter_idx_dict[self.client_index] = self.local_inner_iter_idx

    def update_local_time_info_dict(self, local_time_info_dict):
        self.local_comm_round_idx_dict.update(local_time_info_dict["local_comm_round_idx_dict"])
        self.local_outer_epoch_idx_dict.update(local_time_info_dict["local_outer_epoch_idx_dict"])
        self.local_inner_epoch_idx_dict.update(local_time_info_dict["local_inner_epoch_idx_dict"])
        self.local_outer_iter_idx_dict.update(local_time_info_dict["local_outer_iter_idx_dict"])
        self.local_inner_iter_idx_dict.update(local_time_info_dict["local_inner_iter_idx_dict"])

        if self.client_index in local_time_info_dict["local_comm_round_idx"] and \
            local_time_info_dict["update_local_time_info_dict"] is True:
            logging.warning("WARNING!!! Client {}, receive self time info, Be careful.".format(
                self.client_index
            ))
        else:
            self.refresh_local_time_dict()

    def get_local_time_info(self):
        local_time_info = {}
        local_time_info["client_index"] = self.client_index
        local_time_info["local_comm_round_idx"] = self.local_comm_round_idx
        local_time_info["local_outer_epoch_idx"] = self.local_outer_epoch_idx
        local_time_info["local_inner_epoch_idx"] = self.local_inner_epoch_idx
        local_time_info["local_outer_iter_idx"] = self.local_outer_iter_idx
        local_time_info["local_inner_iter_idx"] = self.local_inner_iter_idx
        return local_time_info

    def get_local_and_global_time_info(self):
        time_info = {}
        time_info["local_time_info"] = self.get_local_time_info()
        time_info["global_time_info"] = self.get_global_time_info()
        return time_info


    def get_time_info_to_send(self, if_send_time_info_dict=False, update_local_time_info_dict=False):
        time_info = {}
        time_info["local_time_info"] = self.get_local_time_info()
        if if_send_time_info_dict:
            time_info["local_time_info_dict"] = self.get_local_time_info_dict(update_local_time_info_dict)
        return time_info

    def update_time_info(self, time_info):
        """
            Remember to revise them if needed.
        """
        if "global_time_info" in time_info:
            self.update_global_time_info(time_info["global_time_info"])
        else:
            raise NotImplementedError

        if "local_time_info" in time_info:
            self.update_local_time_info(time_info["local_time_info"])

        if "local_time_info_dict" in time_info:
            self.update_local_time_info_dict(time_info["local_time_info_dict"])


    def refresh_local_time_dict(self):
        self.local_comm_round_idx_dict[self.client_index] = self.local_comm_round_idx
        self.local_outer_epoch_idx_dict[self.client_index] = self.local_outer_epoch_idx
        self.local_inner_epoch_idx_dict[self.client_index] = self.local_inner_epoch_idx
        self.local_outer_iter_idx_dict[self.client_index] = self.local_outer_iter_idx
        self.local_inner_iter_idx_dict[self.client_index] = self.local_inner_iter_idx


    def past_comm_round(self, comm_round=1):
        self.local_comm_round_idx += comm_round
        self.local_comm_round_idx_dict[self.client_index] = self.local_comm_round_idx


    def past_iterations(self, iterations=1):
        self.local_outer_iter_idx += iterations
        self.local_inner_iter_idx = self.local_outer_iter_idx % self.local_num_iterations
        self.local_outer_epoch_idx = int(self.local_outer_iter_idx / self.local_num_iterations)
        self.local_inner_epoch_idx = self.local_outer_epoch_idx
        self.local_outer_epoch_idx_dict[self.client_index] = self.local_outer_epoch_idx
        self.local_inner_epoch_idx_dict[self.client_index] = self.local_inner_epoch_idx
        self.local_outer_iter_idx_dict[self.client_index] = self.local_outer_iter_idx
        self.local_inner_iter_idx_dict[self.client_index] = self.local_inner_iter_idx


    def past_epochs(self, epochs=1):
        self.local_outer_epoch_idx += epochs
        self.local_inner_epoch_idx = self.local_outer_epoch_idx
        self.local_outer_iter_idx += epochs * self.local_num_iterations
        # self.local_inner_iter_idx
        self.local_outer_epoch_idx_dict[self.client_index] = self.local_outer_epoch_idx
        self.local_inner_epoch_idx_dict[self.client_index] = self.local_inner_epoch_idx
        self.local_outer_iter_idx_dict[self.client_index] = self.local_outer_iter_idx
        # self.local_inner_iter_idx_dict[self.client_index] = self.local_inner_iter_idx


# self.global_comm_round_idx = time_info["global_comm_round_idx"]
# self.local_round_idx = time_info["global_comm_round_idx"]
# self.epoch = self.local_round_idx // self.global_num_iterations
# self.iteration = self.local_round_idx % self.global_num_iterations
# self.total_iteration = self.local_round_idx
