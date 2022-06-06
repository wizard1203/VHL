import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))



from algorithms.basePS.ps_aggregator import PSAggregator


class Aggregator(PSAggregator):
    def __init__(self, train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer=None, metrics=None, traindata_cls_counts=None):
        super().__init__(train_global, test_global, all_train_data_num,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                args, model_trainer, perf_timer, metrics, traindata_cls_counts=traindata_cls_counts)


    def get_max_comm_round(self):
        pass






















