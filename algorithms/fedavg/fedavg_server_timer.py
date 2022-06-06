import logging
from timers.server_timer import ServerTimer



class FedAVGServerTimer(ServerTimer):
    def __init__(self, args, global_num_iterations, local_num_iterations_dict,
                global_epochs_per_round,
                local_num_epochs_per_comm_round_dict):
        super().__init__(args, global_num_iterations, local_num_iterations_dict)
        self.global_epochs_per_round = global_epochs_per_round
        self.local_num_epochs_per_comm_round_dict = local_num_epochs_per_comm_round_dict





    # override
    def past_iterations(self, iterations=1):
        self.global_outer_iter_idx += iterations
        # logging.info(f"self.global_outer_iter_idx: {self.global_outer_iter_idx}, self.global_num_iterations:{self.global_num_iterations}")
        self.global_inner_iter_idx = self.global_outer_iter_idx % self.global_num_iterations
        self.global_outer_epoch_idx = int(self.global_outer_iter_idx / self.global_num_iterations)
        self.global_inner_epoch_idx = self.global_outer_epoch_idx % self.global_epochs_per_round

    # override
    def past_epochs(self, epochs=1):
        self.global_outer_epoch_idx += epochs
        self.global_inner_epoch_idx = self.global_outer_epoch_idx % self.global_epochs_per_round
        self.global_outer_iter_idx += epochs * self.global_num_iterations
        # self.local_inner_iter_idx











