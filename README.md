# VHL


## Code Structure.

The ``algorithms_standalone`` folder includes implementations of FedAvg and other algorithms.
The ``data_preprocessing`` folder includes IID dataloader and non-IID dataloder.
The ``fedml_core`` folder includes the low-level communication core. But for the standalone simulation, we do not need them.
The ``model`` folder includes the architectures of deep neural networks.
The ``experiments`` folder includes the configs of different algorithms and launch files of experiments.

All hyper-parameters are identified in ``configs/default.py``.


## Launch Experiments.

The ``experiments/main_args.conf`` includes the default hyper-parameters.
The ``experiments/configs_algorithm`` includes the configuration hyper-parameters of the specific algorithms.
The ``experiments/configs_system`` includes the configuration hyper-parameters related to the running environment. This is designed for users' convenience of no need to specifying Python Path and Data Dir when launching experiments every time.


To launch experiments, you can ``cd experiments/standlone``. And type ``./launch_standalone.sh`` to run the simple FedAvg code. You can type ``dataset=cifar10 partition_alpha=0.1 partition_method=hetero ./launch_standalone.sh`` to set the dataset as CIFAR-10 and the a=0.1 of LDA partition method.

To launch VHL, you can refer to ``experiments/standlone/VHL_exp/Exp.md`` for more information, in which we offer all scripts of launching baseline and VHL experiments.

## Wandb Usage

If you do not want to use wandb for recording, you can add ``wandb_record=False`` in to the command. Then you don't need to install and config wandb.









