import os
import sys
import random

from yacs.config import CfgNode as _CfgNode



class CfgNode(_CfgNode):

    def setup(self, args):
        self.merge_from_file((args.config_file))
        self.merge_from_list(args.opts)

        # calibrate the path configruration
        if self.model.resume_path:
            self.logger.path = os.path.dirname(self.model.resume_path)
        else:
            if not self.logger.name:
                self.logger.name = 'checkpoint'
            self.logger.version = self._version_logger(self.output_root, self.logger.name)
            self.logger.path = os.path.join(self.output_root, self.logger.name, f'version_{self.logger.version}')
        self.logger.log_file = os.path.join(self.logger.path, 'log.txt')
        cfg_name = os.path.basename(args.config_file)
        self.logger.cfg_file = os.path.join(self.logger.path, cfg_name)
        os.makedirs(self.logger.path, exist_ok=True)
        self.freeze()

        # backup cfg and args
        logger = MyLogger('NAS', self).getlogger()
        logger.info(self)
        logger.info(args)
        with open(self.logger.cfg_file, 'w') as f:
            f.write(str(self))

    @staticmethod
    def _version_logger(save_dir, logger_name=''):
        if logger_name:
            path = os.path.join(save_dir, logger_name)
        else:
            path = save_dir
        if (not os.path.exists(path)) or (not os.listdir(path)):
            version = 0
        else:
            versions = [int(v.split('_')[-1]) for v in os.listdir(path)]
            version = max(versions)+1
        return version

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            v = f"'{v}'" if isinstance(v, str) else v
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 4)
            s.append(attr_str)
        r += "\n".join(s)
        return r

global_cfg = CfgNode()
CN = CfgNode

def get_cfg():
    '''
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    '''
    from .default import _C
    return _C.clone()









_C = CN()
_C.VERSION = 1
_C.config = CN()
_C.config.name = ''


# ---------------------------------------------------------------------------- #
# input
# ---------------------------------------------------------------------------- #

_C.input = CN()
_C.input.size = (224, 224)

# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #
_C.dataset = CN()
_C.dataset.name = 'fakedata'
_C.dataset.datapath = '../datasets'
_C.dataset.trainlist = './datasets/train.txt'
_C.dataset.validlist = './datasets/test.txt'
_C.dataset.testlist = './datasets/test.txt'
_C.dataset.batch_size = 8
_C.dataset.workers = 4
_C.dataset.is_train = False


# ---------------------------------------------------------------------------- #
# transforms
# ---------------------------------------------------------------------------- #

_C.transforms = CN() # image transforms
_C.transforms.name = 'DefaultTransforms'

# albumentations transforms (abtfs)
_C.abtfs = CN()
_C.abtfs.random_grid_shuffle = CN()
_C.abtfs.random_grid_shuffle.enable = 0
_C.abtfs.random_grid_shuffle.grid = 2

_C.abtfs.channel_shuffle = CN()
_C.abtfs.channel_shuffle.enable = 0

_C.abtfs.channel_dropout = CN()
_C.abtfs.channel_dropout.enable = 0
_C.abtfs.channel_dropout.drop_range = (1, 1)
_C.abtfs.channel_dropout.fill_value = 127

_C.abtfs.noise = CN()
_C.abtfs.noise.enable = 1

_C.abtfs.blur = CN()
_C.abtfs.blur.enable = 0

_C.abtfs.rotate = CN()
_C.abtfs.rotate.enable = 1

_C.abtfs.bright = CN()
_C.abtfs.bright.enable = 1

_C.abtfs.distortion = CN()
_C.abtfs.distortion.enable = 0

_C.abtfs.hue = CN()
_C.abtfs.hue.enable = 0

_C.abtfs.cutout = CN()
_C.abtfs.cutout.enable = 1
_C.abtfs.cutout.num_holes = 10
_C.abtfs.cutout.size = 20
_C.abtfs.cutout.fill_value = 127

## transforms for tensor
_C.transforms.tensor = CN()
_C.transforms.tensor.normalization = CN()
_C.transforms.tensor.normalization.mean = [0.5, 0.5, 0.5] 
_C.transforms.tensor.normalization.std = [0.5, 0.5, 0.5] 
_C.transforms.tensor.random_erasing = CN()
_C.transforms.tensor.random_erasing.enable = 0
_C.transforms.tensor.random_erasing.p = 0.5
_C.transforms.tensor.random_erasing.scale = (0.02, 0.3) # range of proportion of erased area against input image.
_C.transforms.tensor.random_erasing.ratio = (0.3, 3.3), # range of aspect ratio of erased area.


## transforms for PIL image
_C.transforms.img = CN()

### modify the image size, only use one operation
# random_resized_crop
_C.transforms.img.random_resized_crop = CN()
_C.transforms.img.random_resized_crop.enable = 0
_C.transforms.img.random_resized_crop.scale = (0.5, 1.0)
_C.transforms.img.random_resized_crop.ratio = (3/4, 4/3)

# resize
_C.transforms.img.resize = CN()
_C.transforms.img.resize.enable = 1

# random_crop
_C.transforms.img.random_crop = CN()
_C.transforms.img.random_crop.enable = 1
_C.transforms.img.random_crop.padding = 0

# center_crop
_C.transforms.img.center_crop = CN()
_C.transforms.img.center_crop.enable = 0

### without modifying the image size
_C.transforms.img.aug_imagenet = False
_C.transforms.img.aug_cifar = False

# color_jitter
_C.transforms.img.color_jitter = CN()
_C.transforms.img.color_jitter.enable = 0
_C.transforms.img.color_jitter.brightness = 0.
_C.transforms.img.color_jitter.contrast = 0.
_C.transforms.img.color_jitter.saturation = 0.
_C.transforms.img.color_jitter.hue = 0.

# horizontal_flip
_C.transforms.img.random_horizontal_flip = CN()
_C.transforms.img.random_horizontal_flip.enable = 1
_C.transforms.img.random_horizontal_flip.p = 0.5

# vertical_flip
_C.transforms.img.random_vertical_flip = CN()
_C.transforms.img.random_vertical_flip.enable = 1
_C.transforms.img.random_vertical_flip.p = 0.5

# random_rotation
_C.transforms.img.random_rotation = CN()
_C.transforms.img.random_rotation.enable = 1
_C.transforms.img.random_rotation.degrees = 10



_C.label_transforms = CN() # label transforms
_C.label_transforms.name = 'default'
# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #
_C.model = CN()
_C.model.name = 'ClsNet'
_C.model.depth = 5 # for u-net structure
_C.model.num_layers = 5 # for classification structure
_C.model.in_channels = 3
_C.model.out_channels = 32
_C.model.expansion = 2
_C.model.num_nodes = 5
_C.model.max_disp = 40 # disparity search range
_C.model.finetune = False
_C.model.resume_path = '' # the resume model path
_C.model.classes = 10
_C.model.use_aux_heads = True
_C.model.dropout_rate = 0.5
_C.model.aux_weight = 0.4


# ---------------------------------------------------------------------------- #
# loss
# ---------------------------------------------------------------------------- #
_C.loss = CN()
_C.loss.name = 'CrossEntropy'  # loss_scheme

_C.loss.CrossEntropy = CN()
_C.loss.CrossEntropy.class_weight = []

# for MultiScaleLoss
_C.loss.MultiScaleLoss = CN()
_C.loss.MultiScaleLoss.sub_loss = 'L1'
_C.loss.MultiScaleLoss.weights = [0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005]
_C.loss.MultiScaleLoss.mask = False
_C.loss.MultiScaleLoss.downscale = 1

# focal loss
_C.loss.focal_loss = CN()
_C.loss.focal_loss.alpha = [2.03316646, 3.4860515 , 5.50677966, 1., 6.33333333, 8.24619289, 3.32889344, 2.75338983, 7.98280098, 8.57255937]
_C.loss.focal_loss.gamma = 2
_C.loss.focal_loss.size_average = True

# ---------------------------------------------------------------------------- #
# optimizer
# ---------------------------------------------------------------------------- #

_C.optim = CN()
_C.optim.name = 'adam'
_C.optim.momentum = 0.9
_C.optim.base_lr = 0.001
_C.optim.weight_decay = 0.0005

# scheduler
_C.optim.scheduler = CN()
_C.optim.scheduler.name = 'CosineAnnealingLR'
_C.optim.scheduler.gamma = 0.1 # decay factor

# for CosineAnnealingLR
_C.optim.scheduler.t_max = 50

# for CosineAnnealingLR
_C.optim.scheduler.t_0 = 5
_C.optim.scheduler.t_mul = 20

# for ReduceLROnPlateau
_C.optim.scheduler.mode = 'min' # min for loss, max for acc
_C.optim.scheduler.patience = 10
_C.optim.scheduler.verbose = True # print log once update lr

# for StepLR
_C.optim.scheduler.step_size = 10

# for MultiStepLR
_C.optim.scheduler.milestones = [10, 25, 35, 50]

# ---------------------------------------------------------------------------- #
# evaluator
# ---------------------------------------------------------------------------- #

_C.evaluator = CN()
_C.evaluator.name = 'DefaultEvaluator'
_C.evaluator.num_epochs = 200


# ---------------------------------------------------------------------------- #
# mutator
# ---------------------------------------------------------------------------- #

_C.mutator = CN()
_C.mutator.name = 'EnasMutator'

_C.mutator.EnasMutator = CN()
_C.mutator.EnasMutator.reward_function = ''
_C.mutator.EnasMutator.lstm_size = 64
_C.mutator.EnasMutator.lstm_num_layers = 1
_C.mutator.EnasMutator.tanh_constant = 1.5
_C.mutator.EnasMutator.cell_exit_extra_step = False
_C.mutator.EnasMutator.skip_target = 0.4
_C.mutator.EnasMutator.branch_bias = 0.25
_C.mutator.EnasMutator.arch_loss_weight = 0.02 # 0.002 small 0.02 medium 0.2 big
_C.mutator.EnasMutator.reward_weight = 50
_C.mutator.EnasMutator.temperature = 2
_C.mutator.EnasMutator.entropy_reduction = 'sum' # 'sum', 'mean'

_C.mutator.DartsMutator = CN()
_C.mutator.DartsMutator.arc_lr = 3e-4
_C.mutator.DartsMutator.unrolled = False # if true: second-order; else: first-order
# ---------------------------------------------------------------------------- #
# trainer
# ---------------------------------------------------------------------------- #

_C.trainer = CN()
_C.trainer.name = 'EnasTrainer'
_C.trainer.startRound = 0
_C.trainer.startEpoch = 0
_C.trainer.num_epochs = 40
_C.trainer.device = 'cuda'
_C.trainer.device_ids = [0]
_C.trainer.warm_start_epoch = 5 # the epoch to warm-start iteratively training model and mutator
_C.trainer.accumulate_steps = 1
_C.trainer.validate_always = False # validate the test set after the training epoch

_C.trainer.EnasTrainer = CN()
_C.trainer.EnasTrainer.entropy_weight = 0.0001
_C.trainer.EnasTrainer.skip_weight = 0.8
_C.trainer.EnasTrainer.baseline_decay = 0.999
_C.trainer.EnasTrainer.mutator_lr = 0.00035
_C.trainer.EnasTrainer.mutator_steps_aggregate = 20
_C.trainer.EnasTrainer.mutator_steps = 50


# ---------------------------------------------------------------------------- #
# callback
# ---------------------------------------------------------------------------- #

_C.callback = CN()
_C.callback.checkpoint = CN()
_C.callback.checkpoint.mode = 'max'
_C.callback.relevance = CN()
_C.callback.relevance.filename = 'relevance.csv'


# ---------------------------------------------------------------------------- #
# training trics
# ---------------------------------------------------------------------------- #
_C.mixup = CN()
_C.mixup.enable = 0
_C.mixup.alpha = 0.4

_C.kd = CN()
_C.kd.enable = 0
_C.kd.model = CN()
_C.kd.model.name = 'Nasnetamobile'
_C.kd.model.path = 'teacher_net.pt'
_C.kd.loss = CN()
_C.kd.loss.alpha = 0.5
_C.kd.loss.temperature = 2


# ---------------------------------------------------------------------------- #
# other
# ---------------------------------------------------------------------------- #
_C.debug = False
_C.comment = ''
def _version_logger(save_dir, logger_name=''):
    if logger_name:
        path = os.path.join(save_dir, logger_name)
    else:
        path = save_dir
    if (not os.path.exists(path)) or (not os.listdir(path)):
        version = 0
    else:
        try:
            versions = [int(v.split('_')[-1]) for v in os.listdir(path)]
            version = max(versions)+1
        except:
            version = 0
    return version

_C.seed = int(random.random()*100)
_C.output_root = './outputs'

_C.logger = CN()
_C.logger.name = ''
_C.logger.version = _version_logger(_C.output_root, _C.logger.name)
_C.logger.path = os.path.join(_C.output_root, 'checkpoint_search', f'version_{_C.logger.version}')
if _C.model.resume_path:
    _C.logger.path = os.path.dirname(_C.model.resume_path)
_C.logger.log_file = os.path.join(_C.logger.path, 'log.txt')
_C.logger.cfg_file = ''
_C.logger.log_frequency = 100 # print log every 'frequency' steps
