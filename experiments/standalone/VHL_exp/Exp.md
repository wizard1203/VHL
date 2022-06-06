
## 

You can firstly ``cd experiments/standalone``, and use ``VHL_exp/run.sh`` to launch experiments. You can refer to following scripts to launch experiments.


## CIFAR10 

wandb_record=False \
gpu_index=3 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=fedavg   bash VHL_exp/run.sh

wandb_record=False \
gpu_index=0 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=fedavg  fedprox=True   bash VHL_exp/run.sh

gpu_index=1 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=fedavg scaffold=True   bash VHL_exp/run.sh

gpu_index=2 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova  bash VHL_exp/run.sh


wandb_record=False \
gpu_index=1 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast   bash VHL_exp/run.sh

gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  fedprox=True   bash VHL_exp/run.sh

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast scaffold=True   bash VHL_exp/run.sh

gpu_index=3 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar10 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  algorithm=FedNova  bash VHL_exp/run.sh


## Fmnist
gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=fmnist_normal client_number_conf=client10 algo_conf=fedavg   bash VHL_exp/run.sh

gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=fmnist_normal client_number_conf=client10 algo_conf=fedavg  fedprox=True \
bash VHL_exp/run.sh

gpu_index=1 partition_alpha=0.1 cluster_conf=localhost task_conf=fmnist_normal client_number_conf=client10 algo_conf=fedavg  scaffold=True \
bash VHL_exp/run.sh

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=fmnist_normal client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova \
bash VHL_exp/run.sh


wandb_record=False \
gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=fmnist_withnoise client_number_conf=client10 algo_conf=VHL_alignNoiseContrast   bash VHL_exp/run.sh

gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=fmnist_withnoise client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  fedprox=True   bash VHL_exp/run.sh

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=fmnist_withnoise client_number_conf=client10 algo_conf=VHL_alignNoiseContrast scaffold=True   bash VHL_exp/run.sh

gpu_index=3 partition_alpha=0.1  cluster_conf=localhost task_conf=fmnist_withnoise client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  algorithm=FedNova  bash VHL_exp/run.sh





## SVHN

gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=fedavg \
bash VHL_exp/run.sh


gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=fedavg fedprox=True \
  bash VHL_exp/run.sh

gpu_index=3 partition_alpha=0.1 cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=fedavg scaffold=True lr=0.001 \
bash VHL_exp/run.sh

gpu_index=3 partition_alpha=0.1 cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova \
 bash VHL_exp/run.sh



wandb_record=False \
gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=VHL_alignNoiseContrast   bash VHL_exp/run.sh

gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  fedprox=True   bash VHL_exp/run.sh

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=VHL_alignNoiseContrast scaffold=True   bash VHL_exp/run.sh

gpu_index=3 partition_alpha=0.1  cluster_conf=localhost task_conf=SVHN client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  algorithm=FedNova  bash VHL_exp/run.sh




## CIFAR-100


gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=fedavg \
bash VHL_exp/run.sh

gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=fedavg fedprox=True \
bash VHL_exp/run.sh


gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=fedavg scaffold=True \
bash VHL_exp/run.sh


gpu_index=1 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=fedavg  algorithm=FedNova \
bash VHL_exp/run.sh



wandb_record=False \
gpu_index=0 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast   bash VHL_exp/run.sh

gpu_index=1 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  fedprox=True   bash VHL_exp/run.sh

gpu_index=2 partition_alpha=0.1 cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast scaffold=True   bash VHL_exp/run.sh

gpu_index=3 partition_alpha=0.1  cluster_conf=localhost task_conf=cifar100 client_number_conf=client10 algo_conf=VHL_alignNoiseContrast  algorithm=FedNova  bash VHL_exp/run.sh























































