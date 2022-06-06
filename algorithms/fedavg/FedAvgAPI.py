from mpi4py import MPI

from utils.perf_timer_with_cuda import Perf_Timer
from utils.metrics import Metrics
from utils.logger import Logger

from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager

track_time = True

def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer=None):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                    model_trainer)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict, test_data_local_dict, model_trainer)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, model_trainer):
    model_trainer.set_id(-1)

    perf_timer = Perf_Timer(
        verbosity_level=1 if track_time else 0,
        log_fn=Logger.log_timer
    )
    metrics = Metrics([1], task=args.task)

    # aggregator
    worker_num = size - 1
    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num,
                                train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                worker_num, device, args, model_trainer, perf_timer, metrics)

    # start the distributed training
    server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, "MPI", perf_timer, metrics)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict,
                train_data_local_dict, test_data_local_dict, model_trainer=None):
    client_index = process_id - 1
    model_trainer.set_id(client_index)

    perf_timer = Perf_Timer(
        verbosity_level=1 if track_time else 0,
        log_fn=Logger.log_timer
    )
    metrics = Metrics([1], task=args.task)

    trainer = FedAVGTrainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                            train_data_num, device, args, model_trainer, perf_timer, metrics)
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size, "MPI", perf_timer, metrics)
    client_manager.run()
