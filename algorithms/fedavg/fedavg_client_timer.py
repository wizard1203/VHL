from timers.client_timer import ClientTimer


class FedAVGClientTimer(ClientTimer):
    def __init__(self, args, global_num_iterations, local_num_iterations_dict,
                global_epochs_per_round,
                local_num_epochs_per_comm_round_dict, client_index=None):
        super().__init__(args, global_num_iterations, local_num_iterations_dict, client_index)
        self.global_epochs_per_round = global_epochs_per_round
        self.local_num_epochs_per_comm_round_dict = local_num_epochs_per_comm_round_dict
        self.local_num_epochs_per_comm_round = local_num_epochs_per_comm_round_dict[self.client_index]


    # override
    def past_iterations(self, iterations=1):
        self.local_outer_iter_idx += iterations
        self.local_inner_iter_idx = self.local_outer_iter_idx % self.local_num_iterations
        self.local_outer_epoch_idx = int(self.local_outer_iter_idx / self.local_num_iterations)
        self.local_inner_epoch_idx = self.local_outer_epoch_idx % self.local_num_epochs_per_comm_round
        self.local_outer_epoch_idx_dict[self.client_index] = self.local_outer_epoch_idx
        self.local_inner_epoch_idx_dict[self.client_index] = self.local_inner_epoch_idx
        self.local_outer_iter_idx_dict[self.client_index] = self.local_outer_iter_idx
        self.local_inner_iter_idx_dict[self.client_index] = self.local_inner_iter_idx


    # override
    def past_epochs(self, epochs=1):
        self.local_outer_epoch_idx += epochs
        self.local_inner_epoch_idx = self.local_outer_epoch_idx % self.local_num_epochs_per_comm_round
        self.local_outer_iter_idx += epochs * self.local_num_iterations
        # self.local_inner_iter_idx
        self.local_outer_epoch_idx_dict[self.client_index] = self.local_outer_epoch_idx
        self.local_inner_epoch_idx_dict[self.client_index] = self.local_inner_epoch_idx
        self.local_outer_iter_idx_dict[self.client_index] = self.local_outer_iter_idx
        # self.local_inner_iter_idx_dict[self.client_index] = self.local_inner_iter_idx






