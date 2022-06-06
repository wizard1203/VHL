import logging

from .base_timer import Timer


class ServerTimer(Timer):
    def __init__(self, args, global_num_iterations, local_num_iterations_dict):
        super().__init__(args)
        self.role = 'server'
        self.server_index = args.server_index
        self.global_num_iterations = global_num_iterations
        self.local_num_iterations_dict = local_num_iterations_dict


    def update_local_time_info(self, local_time_info):
        client_index = local_time_info["client_index"]
        self.local_comm_round_idx_dict[client_index] = local_time_info["local_comm_round_idx"]
        self.local_outer_epoch_idx_dict[client_index] = local_time_info["local_outer_epoch_idx"]
        self.local_inner_epoch_idx_dict[client_index] = local_time_info["local_inner_epoch_idx"]
        self.local_outer_iter_idx_dict[client_index] = local_time_info["local_outer_iter_idx"]
        self.local_inner_iter_idx_dict[client_index] = local_time_info["local_inner_iter_idx"]


    def update_time_info(self, time_info):
        """
            Remember to revise them if needed.
        """
        # if "global_time_info" in time_info:
        #     self.update_global_time_info(time_info["global_time_info"])

        if "local_time_info" in time_info:
            self.update_local_time_info(time_info["local_time_info"])
        else:
            raise NotImplementedError

        if "local_time_info_dict" in time_info:
            self.update_local_time_info_dict(time_info["local_time_info_dict"])


    def get_time_info_to_send(self, if_send_time_info_dict=False, update_local_time_info_dict=False):
        time_info = {}
        time_info["global_time_info"] = self.get_global_time_info()
        if if_send_time_info_dict:
            time_info["local_time_info_dict"] = self.get_local_time_info_dict(update_local_time_info_dict)
        return time_info

    def past_comm_round(self, comm_round=1):
        self.global_comm_round_idx += comm_round


    def past_iterations(self, iterations=1):
        self.global_outer_iter_idx += iterations
        self.global_inner_iter_idx = self.global_outer_iter_idx % self.global_num_iterations
        self.global_outer_epoch_idx = int(self.global_outer_iter_idx / self.global_num_iterations)
        self.global_inner_epoch_idx = self.global_outer_epoch_idx

    def past_epochs(self, epochs=1):
        self.global_outer_epoch_idx += epochs
        self.global_inner_epoch_idx = self.global_outer_epoch_idx
        self.global_outer_iter_idx += epochs * self.global_num_iterations
        # self.local_inner_iter_idx
