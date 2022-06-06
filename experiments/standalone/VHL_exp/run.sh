#!/bin/bash

algo_conf=${algo_conf:-fednoise_alignClass}
client_number_conf=${client_number_conf:-client10}
cluster_conf=${cluster_conf:-localhost}
task_conf=${task_conf:-cifar10}


export data_save_memory_mode=${data_save_memory_mode:-False}

source VHL_exp/client_number/${client_number_conf}.sh
source VHL_exp/clusters/${cluster_conf}.sh
source VHL_exp/tasks/${task_conf}.sh
source VHL_exp/algorithms/${algo_conf}.sh


export cluster_name=${cluster_name:-localhost}


# export WANDB_MODE=offline

export entity="coolzhtang"
export project="VHL"
export level=${level:-INFO}
export exp_mode=${exp_mode:-"ready"}

export gpu_index=${gpu_index:-0}



# export sched="StepLR"
# # export lr_decay_rate=0.97
# export lr_decay_rate=0.992
# export momentum=0.9
export sched=${sched:-StepLR}
# export lr_decay_rate=0.97
export lr_decay_rate=${lr_decay_rate:-0.992}
export momentum=${momentum:-0.9}
export global_epochs_per_round=${global_epochs_per_round:-1}
export max_epochs=${max_epochs:-1000}

export batch_size=${batch_size:-128}
export wd=${wd:-0.0001}

export client_num_in_total=${client_num_in_total:-10}
export client_num_per_round=${client_num_per_round:-5}


export partition_method=${partition_method:-'hetero'}
# export partition_alpha=0.05
export partition_alpha=${partition_alpha:-0.1}

export fedprox_mu=${fedprox_mu:-1.0}


bash ./launch_standalone.sh








