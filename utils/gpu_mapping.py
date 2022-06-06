import logging
import socket
import yaml

import torch


def init_training_device_from_gpu_util_file(process_id, worker_number, gpu_util_file, gpu_util_key):

    if gpu_util_file == None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ##################  Not Indicate gpu_util_file, using cpu  #################")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device
    else:
        with open(gpu_util_file, 'r') as f:
            gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
            # gpu_util_num_process = 'gpu_util_' + str(worker_number)
            # gpu_util = gpu_util_yaml[gpu_util_num_process]
            gpu_util = gpu_util_yaml[gpu_util_key]
            gpu_util_map = {}
            i = 0
            for host, gpus_util_map_host in gpu_util.items():
                for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                    for _ in range(num_process_on_gpu):
                        gpu_util_map[i] = (host, gpu_j)
                        i += 1
            logging.info("Process %d running on host: %s,gethostname: %s, gpu: %d ..." % (
                process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
            assert i == worker_number

        device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device, gpu_util_map

def init_training_device_from_gpu_util_parse(process_id, worker_number, gpu_util_parse):
    if gpu_util_parse == None:
        device = torch.device("cpu")
        logging.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(" ##################  Not Indicate gpu_util_file, using cpu  #################")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device
    else:
        # example parse str `gpu_util_parse`: 
        # "gpu1:0,1,1,2;gpu2:3,3,3;gpu3:0,0,0,1,2,4,4,0"
        gpu_util_parse_temp = gpu_util_parse.split(';')
        gpu_util_parse_temp = [(item.split(':')[0], item.split(':')[1]) for item in gpu_util_parse_temp ]

        gpu_util = {}
        for (host, gpus_str) in gpu_util_parse_temp:
            gpu_util[host] = [int(num_process_on_gpu) for num_process_on_gpu in gpus_str.split(',')]

        gpu_util_map = {}
        i = 0
        for host, gpus_util_map_host in gpu_util.items():
            for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                for _ in range(num_process_on_gpu):
                    gpu_util_map[i] = (host, gpu_j)
                    i += 1
        logging.info("Process %d running on host: %s,gethostname: %s, gpu: %d ..." % (
            process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
        assert i == worker_number

        device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
        logging.info(device)
        #return gpu_util_map[process_id][1]
        return device, gpu_util_map



