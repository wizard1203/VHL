import copy
import logging
import time

import torch
import wandb
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from torch.distributions import Categorical

from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from fedml_core.trainer.model_trainer import ModelTrainer

from data_preprocessing.utils.stats import record_batch_data_stats

from utils.data_utils import (
    get_data,
    get_named_data,
    get_all_bn_params,
    apply_gradient,
    clear_grad,
    get_name_params_difference,
    get_local_num_iterations,
    get_avg_num_iterations,
    check_device,
    get_train_batch_data
)

from utils.model_utils import (
    set_freeze_by_names,
    get_actual_layer_names,
    freeze_by_names,
    unfreeze_by_names,
    get_modules_by_names
)

from utils.matrix_utils import orthogo_tensor

from utils.distribution_utils import train_distribution_diversity

from utils.context import (
    raise_error_without_process,
    get_lock,
)

from utils.checkpoint import (
    setup_checkpoint_config, setup_save_checkpoint_path, save_checkpoint,
    setup_checkpoint_file_name_prefix,
    save_checkpoint_without_check
)


from model.build import create_model
from data_preprocessing.build import (
    VHL_load_dataset
)

from trainers.averager import Averager
from trainers.tSNE import Dim_Reducer

from loss_fn.cov_loss import (
    cov_non_diag_norm, cov_norm
)
from loss_fn.losses import LabelSmoothingCrossEntropy, proxy_align_loss, align_feature_loss


class NormalTrainer(ModelTrainer):
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        super().__init__(model)

        if kwargs['role'] == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif kwargs['role'] == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError

        self.role = kwargs['role']

        self.args = args
        self.model = model
        # self.model.to(device)
        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

        self.save_checkpoints_config = setup_checkpoint_config(self.args)

        # For future use
        self.param_groups = self.optimizer.param_groups
        with raise_error_without_process():
            self.param_names = list(
                enumerate([group["name"] for group in self.param_groups])
            )

        self.named_parameters = list(self.model.named_parameters())

        if len(self.named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                    in sorted(self.named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'noname.%s' % i
                                    for param_group in self.param_groups
                                    for i, v in enumerate(param_group['params'])}

        self.averager = Averager(self.args, self.model)

        self.lr_scheduler = lr_scheduler


        if self.args.VHL:
            if self.args.VHL_feat_align:
                if self.args.VHL_inter_domain_ortho_mapping:
                    self.VHL_mapping_matrix = None
                else:
                    self.VHL_mapping_matrix = torch.rand(
                        self.args.model_feature_dim, self.args.model_feature_dim)
                self.proxy_align_loss = proxy_align_loss(
                    inter_domain_mapping=self.args.VHL_inter_domain_mapping,
                    inter_domain_class_match=self.args.VHL_class_match,
                    noise_feat_detach=self.args.VHL_feat_detach,
                    noise_contrastive=self.args.VHL_noise_contrastive,
                    inter_domain_mapping_matrix=self.VHL_mapping_matrix,
                    inter_domain_weight=self.args.VHL_feat_align_inter_domain_weight,
                    inter_class_weight=self.args.VHL_feat_align_inter_cls_weight,
                    noise_supcon_weight=self.args.VHL_noise_supcon_weight,
                    noise_label_shift=self.args.num_classes,
                    device=self.device)
            if self.args.VHL_data == 'dataset':
                if self.args.VHL_dataset_from_server:
                    self.train_generative_dl_dict = {}
                    self.test_generative_dl_dict = {}
                    self.train_generative_ds_dict = {}
                    self.test_generative_ds_dict = {}
                    self.dataset_label_shift = {}

                    self.train_generative_iter_dict = {}
                    self.test_generative_iter_dict = {}

                else:
                    self.create_noise_dataset_dict()
            else:
                raise NotImplementedError

        if self.args.fed_align:
            self.feature_align_means = torch.rand(
                self.args.num_classes, self.args.model_feature_dim
            )
            self.align_feature_loss = align_feature_loss(
                self.feature_align_means, self.args.fed_align_std, device
            )
    def track(self, tracker, summary_n_samples, model, loss, end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):
        pass

    def epoch_init(self):
        pass

    def epoch_end(self):
        pass

    def update_state(self, **kwargs):
        # This should be called begin the training of each epoch.
        self.update_loss_state(**kwargs)
        self.update_optimizer_state(**kwargs)

    def update_loss_state(self, **kwargs):
        if self.args.loss_fn in ["LDAMLoss", "FocalLoss"]:
            kwargs['cls_num_list'] = kwargs["selected_cls_num_list"]
            self.criterion.update(**kwargs)
        elif self.args.loss_fn in ["local_FocalLoss", "local_LDAMLoss"]:
            kwargs['cls_num_list'] = kwargs["local_cls_num_list_dict"][self.index]
            self.criterion.update(**kwargs)

    def update_optimizer_state(self, **kwargs):
        pass


    def generate_fake_data(self, num_of_samples=64):
        input = torch.randn(num_of_samples, self.args.model_input_channels,
                    self.args.dataset_load_image_size, self.args.dataset_load_image_size)
        return input


    def get_model_named_modules(self):
        return dict(self.model.cpu().named_modules())


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # for name, param in model_parameters.items():
        #     logging.info(f"Getting params as model_parameters: name:{name}, shape: {param.shape}")
        self.model.load_state_dict(model_parameters)


    def set_VHL_mapping_matrix(self, VHL_mapping_matrix):
        self.VHL_mapping_matrix = VHL_mapping_matrix
        self.proxy_align_loss.inter_domain_mapping_matrix = VHL_mapping_matrix

    def get_VHL_mapping_matrix(self):
        return self.VHL_mapping_matrix


    def set_feature_align_means(self, feature_align_means):
        self.feature_align_means = feature_align_means
        self.align_feature_loss.feature_align_means = feature_align_means

    def get_feature_align_means(self):
        return self.feature_align_means
    def create_noise_dataset_dict(self):
        self.train_generative_dl_dict, self.test_generative_dl_dict, \
        self.train_generative_ds_dict, self.test_generative_ds_dict \
            = VHL_load_dataset(self.args)
        self.noise_dataset_label_shift = {}
        noise_dataset_label_init = 0
        next_label_shift = noise_dataset_label_init
        for dataset_name in self.train_generative_dl_dict.keys():
            self.noise_dataset_label_shift[dataset_name] = next_label_shift
            next_label_shift += next_label_shift + self.train_generative_ds_dict[dataset_name].class_num

        if self.args.generative_dataset_shared_loader:
            # These two dataloader iters are shared
            self.train_generative_iter_dict = {}
            self.test_generative_iter_dict = {}

    def generate_orthogonal_random_matrix(self):
        logging.info(f"Generating orthogonal_random_matrix, Calculating.............")
        self.VHL_mapping_matrix = orthogo_tensor(
            raw_dim=self.args.model_feature_dim,
            column_dim=self.args.model_feature_dim)
        logging.info(f"Generating orthogonal_random_matrix, validating: det of matrix: "+
                    f"{torch.det(self.VHL_mapping_matrix)}")

    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 


    def get_model_grads(self):
        named_grads = get_named_data(self.model, mode='GRAD', use_cuda=True)
        # logging.info(f"Getting grads as named_grads: {named_grads}")
        return named_grads

    def set_grad_params(self, named_grads):
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device))


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        return self.optimizer.state


    def clear_optim_buffer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()


    def lr_schedule(self, progress):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(progress)
        else:
            logging.info("No lr scheduler...........")


    def warmup_lr_schedule(self, iterations):
        if self.lr_scheduler is not None:
            self.lr_scheduler.warmup_step(iterations)

    # Used for single machine training
    # Should be discarded #TODO
    def train(self, train_data, device, args, **kwargs):
        model = self.model

        model.train()

        epoch_loss = []
        for epoch in range(args.max_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()

                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                logging.info('Training Epoch: {} iter: {} \t Loss: {:.6f}'.format(
                                epoch, batch_idx, loss.item()))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Train Epo: {} \tLoss: {:.6f}'.format(
                    self.index, epoch, sum(epoch_loss) / len(epoch_loss)))
            self.lr_scheduler.step(epoch=epoch + 1)


    def get_train_batch_data(self, train_local):
        try:
            train_batch_data = self.train_local_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data


    def summarize(self, model, output, labels,
        tracker, metrics,
        loss,
        epoch, batch_idx,
        mode='train',
        checkpoint_extra_name="centralized",
        things_to_track=[],
        if_update_timer=False,
        train_data=None, train_batch_data=None,
        end_of_epoch=None,
    ):
        # if np.isnan(loss.item()):
        # logging
        if np.isnan(loss):
            logging.info('(WARNING!!!!!!!! Trainer_ID {}. Train epoch: {},\
                iteration: {}, loss is nan!!!! '.format(
                self.index, epoch, batch_idx))
            # loss.data.fill_(100)
            loss = 100
        metric_stat = metrics.evaluate(loss, output, labels)
        tracker.update_metrics(
            metric_stat, 
            metrics_n_samples=labels.size(0)
        )

        if len(things_to_track) > 0:
            if end_of_epoch is not None:
                pass
            else:
                end_of_epoch = (batch_idx == len(train_data) - 1)
            self.track(tracker, self.args.batch_size, model, loss, end_of_epoch,
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track)

        if if_update_timer:
            """
                Now, the client timer need to be updated by each iteration, 
                so as to record the track infomation.
                But only for epoch training, because One-step training will be scheduled by client or server
            """
            tracker.timer.past_iterations(iterations=1)

        if mode == 'train':
            logging.info('Trainer {}. Glob comm round: {}, Train Epo: {}, iter: {} '.format(
                self.index, tracker.timer.global_comm_round_idx, epoch, batch_idx) + metrics.str_fn(metric_stat))
                # logging.info('(Trainer_ID {}. Local Training Epoch: {}, Iter: {} \tLoss: {:.6f} ACC1:{}'.format(
                #     self.index, epoch, batch_idx, sum(batch_loss) / len(batch_loss), metric_stat['Acc1']))
        elif mode == 'test':
            logging.info('(Trainer_ID {}. Test epoch: {}, iteration: {} '.format(
                self.index, epoch, batch_idx) + metrics.str_fn(metric_stat))
        else:
            raise NotImplementedError
        return metric_stat



    def train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        model = self.model

        if move_to_gpu:
            model.to(device)
        model.train()
        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data

            real_batch_size = labels.shape[0]
            if self.args.TwoCropTransform:
                x = torch.cat([x[0], x[1]], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x, labels = x.to(device), labels.to(device)
            if clear_grad_bef_opt:
                self.optimizer.zero_grad()

            if self.args.model_out_feature:
                output, feat = model(x)
            else:
                output = model(x)

            loss = self.criterion(output, labels)

            if self.args.fed_align:
                loss_align_Gaussianfeature = self.align_feature_loss(feat, labels, real_batch_size)
                loss += self.args.fed_align_alpha * loss_align_Gaussianfeature
                tracker.update_local_record(
                        'losses_track',
                        server_index=self.server_index,
                        client_index=self.client_index,
                        summary_n_samples=real_batch_size*1,
                        args=self.args,
                        loss_align_Gaussfeat=loss_align_Gaussianfeature.item()
                    )
            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg

            loss.backward()
            loss_value = loss.item()
            self.optimizer.step()

            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                else:
                    current_lr = self.args.lr
                for name, param in model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - current_lr * \
                        check_device((c_model_global[name] - c_model_local[name]), param.data.device)

            logging.debug(f"epoch: {epoch}, Loss is {loss_value}")

            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(model, output, labels,
                        tracker, metrics,
                        loss_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )



    def generate_noise_data(self, noise_label_style="extra", y_train=None):
        # means = torch.zeros((self.args.batch_size, self.args.fedaux_noise_size))
        # noise = torch.normal(mean=means, std=1.0)

        noise_label_shift = 0

        if noise_label_style == "extra":
            noise_label_shift = self.args.num_classes
            chunk_num = self.args.VHL_num
            chunk_size = self.args.batch_size // chunk_num
            # chunks = np.ones(chunk_num)* chunk_size
            chunks = [chunk_size] * chunk_num
            for i in range(self.args.batch_size - chunk_num * chunk_size):
                chunks[i] += 1
        elif noise_label_style == "patch":
            noise_label_shift = 0
            if self.args.VHL_data == "dataset" and self.args.VHL_label_from == "dataset":
                chunk_num = self.args.num_classes
                bs = y_train.shape[0]
                batch_cls_counts = record_batch_data_stats(y_train, bs=bs, num_classes=self.args.num_classes)
                inverse_label_weights = [ bs / (num_label+1)  for label, num_label in batch_cls_counts.items()]
                sum_weights = sum(inverse_label_weights)

                noise_label_weights = [ label_weight / sum_weights for i, label_weight in enumerate(inverse_label_weights)]
                chunks = [int(noise_label_weight *self.args.batch_size)  for noise_label_weight in noise_label_weights]
            else:
                pass
        else:
            raise NotImplementedError

        if self.args.VHL_data == "dataset" and self.args.VHL_label_from == "dataset":

            noise_data_list = []
            noise_data_labels = []
            # In order to implement traverse the extra datasets, automatically generate iterator.
            for dataset_name, train_generative_dl in self.train_generative_dl_dict.items():
                # train_batch_data = get_train_batch_data(self.train_generative_iter_dict, dataset_name,
                #     train_generative_dl, batch_size=self.args.batch_size / len(self.train_generative_dl_dict))
                train_batch_data = get_train_batch_data(self.train_generative_iter_dict, dataset_name,
                    train_generative_dl,
                    batch_size=self.args.VHL_dataset_batch_size / len(self.train_generative_dl_dict))
                logging.debug(f"id(self.train_generative_iter_dict) : {id(self.train_generative_iter_dict)}")
                data, label = train_batch_data
                # logging.debug(f"data.shape: {data.shape}")
                noise_data_list.append(data)
                label_shift = self.noise_dataset_label_shift[dataset_name] + noise_label_shift
                noise_data_labels.append(label + label_shift)
            noise_data = torch.cat(noise_data_list).to(self.device)
            labels = torch.cat(noise_data_labels).to(self.device)
        else:
            raise NotImplementedError

        if self.args.VHL_data_re_norm:
            noise_data = noise_data / 0.25 * 0.5

        return noise_data, labels


    def VHL_train_generator(self):
        for i in range(self.args.VHL_generator_num):
            generator = self.generator_dict[i]
            self.train_generator_diversity(generator, 50)
            # self.train_generator_diversity(generator, 5)


    def train_generator_diversity(self, generator, max_iters=100, min_loss=0.0):
        generator.train()
        generator.to(self.device)
        for i in range(max_iters):
            generator_optimizer = torch.optim.SGD(generator.parameters(),
                lr=0.01, weight_decay=0.0001, momentum=0.9)
            means = torch.zeros((64, self.args.fedaux_noise_size))
            z = torch.normal(mean=means, std=1.0).to(self.device)
            data = generator(z)
            loss_diverse = cov_non_diag_norm(data)
            generator_optimizer.zero_grad()
            loss_diverse.backward()
            generator_optimizer.step()
            logging.info(f"Iteration: {i}, loss_diverse: {loss_diverse.item()}")
            if loss_diverse.item() < min_loss:
                logging.info(f"Iteration: {i}, loss_diverse: {loss_diverse.item()} smaller than min_loss: {min_loss}, break")
                break
        generator.cpu()


    def VHL_get_diverse_distribution(self):
        n_dim = self.generator_dict[0].num_layers
        normed_n_mean = train_distribution_diversity(
            n_distribution=self.args.VHL_num, n_dim=n_dim, max_iters=500)
        self.style_GAN_latent_noise_mean = normed_n_mean.detach()
        self.style_GAN_latent_noise_std = [0.1 / n_dim]*n_dim

        global_zeros = torch.ones((self.args.VHL_num, self.args.style_gan_style_dim)) * 0.0
        global_mean_vector = torch.normal(mean=global_zeros, std=self.args.style_gan_sample_z_mean)
        self.style_GAN_sample_z_mean = global_mean_vector
        self.style_GAN_sample_z_std = self.args.style_gan_sample_z_std


    def VHL_train_one_epoch(self, train_data=None, device=None, args=None, epoch=0,
                        tracker=None, metrics=None,
                        local_iterations=None,
                        move_to_gpu=True, make_summary=True,
                        clear_grad_bef_opt=True, clear_grad_aft_opt=True,
                        hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
                        grad_per_sample=False,
                        fim_tr=False, fim_tr_emp=False,
                        parameters_crt_names=[],
                        checkpoint_extra_name="centralized",
                        things_to_track=[],
                        **kwargs):
        model = self.model

        time_table = {}

        if move_to_gpu:
            # self.generator.to(device)
            model.to(device)

        # self.generator.train()
        model.train()
        # self.generator.eval()

        batch_loss = []
        if local_iterations is None:
            iterations = len(train_data)
        else:
            iterations = local_iterations

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]

        # for batch_idx, (x, labels) in enumerate(train_data):
        for batch_idx in range(iterations):
            train_batch_data = self.get_train_batch_data(train_data)
            x, labels = train_batch_data
            # if batch_idx > 5:
            #     break
            x, labels = x.to(device), labels.to(device)
            if clear_grad_bef_opt:
                self.optimizer.zero_grad()

            time_now = time.time()
            real_batch_size = labels.shape[0]

            aux_data, sampled_label = self.generate_noise_data(
                noise_label_style=self.args.VHL_label_style, y_train=labels)
            sampled_label = sampled_label.to(device)
            if x.shape[1] == 1:
                assert self.args.dataset in ["mnist", "femnist", "fmnist", "femnist-digit"]
                x = x.repeat(1, 3, 1, 1)
            # logging.info(f"x.shape: {x.shape}, aux_data.shape: {aux_data.shape} " )
            x_cat = torch.cat((x, aux_data), dim=0)

            # y_cat = torch.cat((labels, labels), dim=0)
            if self.args.model_out_feature:
                output, feat = model(x_cat)
            else:
                output = model(x_cat)

            loss_origin = F.cross_entropy(output[0:real_batch_size], labels)
            loss_aux = F.cross_entropy(output[real_batch_size:], sampled_label)
            # loss = (1 - alpha) * loss_origin + alpha * loss_aux
            loss = loss_origin + self.args.VHL_alpha * loss_aux
            loss_origin_value = loss_origin.item()

            align_domain_loss_value = 0.0
            align_cls_loss_value = 0.0
            noise_cls_loss_value = 0.0
            if self.args.VHL_feat_align and epoch < self.args.VHL_align_local_epoch:
                loss_feat_align, align_domain_loss_value, align_cls_loss_value, noise_cls_loss_value = self.proxy_align_loss(
                    feat, torch.cat([labels, sampled_label], dim=0), real_batch_size)
                loss += loss_feat_align

            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg

            # loss = F.cross_entropy(output, y_cat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                else:
                    current_lr = self.args.lr
                for name, param in model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - current_lr * \
                        check_device((c_model_global[name] - c_model_local[name]), param.data.device)

            logging.debug(f"epoch: {epoch}, Loss is {loss.item()}")

            metric_stat = metrics.evaluate(loss_aux, output[real_batch_size:], sampled_label)
            tracker.update_local_record(
                    'generator_track',
                    server_index=self.server_index,
                    client_index=self.client_index,
                    summary_n_samples=real_batch_size*1,
                    args=self.args,
                    Loss=metric_stat["Loss"],
                    Acc1=metric_stat["Acc1"],
                    align_domain_loss_value=align_domain_loss_value,
                    align_cls_loss_value=align_cls_loss_value,
                    noise_cls_loss_value=noise_cls_loss_value,
                )


            if make_summary and (tracker is not None) and (metrics is not None):
                self.summarize(model, output[0:real_batch_size], labels,
                        tracker, metrics,
                        loss_origin_value,
                        epoch, batch_idx,
                        mode='train',
                        checkpoint_extra_name=checkpoint_extra_name,
                        things_to_track=things_to_track,
                        if_update_timer=True if self.args.record_dataframe else False,
                        train_data=train_data, train_batch_data=train_batch_data,
                        end_of_epoch=None,
                    )
        return loss, output, labels, x_cat





    def train_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
            grad_per_sample=False,
            fim_tr=False, fim_tr_emp=False,
            parameters_crt_names=[],
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):

        model = self.model

        if move_to_gpu:
            model.to(device)

        model.train()

        x, labels = train_batch_data

        if self.args.TwoCropTransform:
            x = torch.cat([x[0], x[1]], dim=0)
            labels = torch.cat([labels, labels], dim=0)

        x, labels = x.to(device), labels.to(device)
        real_batch_size = labels.shape[0]

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        if self.args.fedprox:
            # previous_model = copy.deepcopy(self.trainer.get_model_params())
            previous_model = kwargs["previous_model"]
        else:
            previous_model = None

        if self.args.VHL:
            aux_data, sampled_label = self.generate_noise_data(
                noise_label_style=self.args.VHL_label_style, y_train=labels)
            # sampled_label = torch.full((self.args.batch_size), self.args.num_classes).long()
            # sampled_label = (torch.ones(self.args.batch_size)*self.args.num_classes).long().to(device)
            sampled_label = sampled_label.to(device)
            # self.generator.eval()

            if x.shape[1] == 1:
                assert self.args.dataset in ["mnist", "femnist", "fmnist", "femnist-digit"]
                x = x.repeat(1, 3, 1, 1)

            x_cat = torch.cat((x, aux_data), dim=0)

            if self.args.model_out_feature:
                output, feat = model(x_cat)
            else:
                output = model(x_cat)
            loss_origin = F.cross_entropy(output[0:real_batch_size], labels)
            loss_aux = F.cross_entropy(output[real_batch_size:], sampled_label)
            # loss = (1 - alpha) * loss_origin + alpha * loss_aux
            loss = loss_origin + self.args.VHL_alpha * loss_aux
            loss_origin_value = loss_origin.item()

            align_domain_loss_value = 0.0
            align_cls_loss_value = 0.0
            noise_cls_loss_value = 0.0
            if self.args.VHL_feat_align and epoch < self.args.VHL_align_local_epoch:
                loss_feat_align, align_domain_loss_value, align_cls_loss_value, noise_cls_loss_value = self.proxy_align_loss(
                    feat, torch.cat([labels, sampled_label], dim=0), real_batch_size)
                loss += loss_feat_align
        else:
            output = model(x)
        loss = self.criterion(output, labels)

        if self.args.fed_align:
            loss_align_Gaussianfeature = self.align_feature_loss(feat, labels, real_batch_size)
            loss += self.args.fed_align_alpha * loss_align_Gaussianfeature
            tracker.update_local_record(
                    'losses_track',
                    server_index=self.server_index,
                    client_index=self.client_index,
                    summary_n_samples=real_batch_size*1,
                    args=self.args,
                    loss_align_Gaussfeat=loss_align_Gaussianfeature.item()
                )

        loss.backward()
        loss_value = loss.item()

        self.optimizer.step()

        if make_summary and (tracker is not None) and (metrics is not None):
            # logging.info(f"")
            self.summarize(model, output, labels,
                    tracker, metrics,
                    loss_value,
                    epoch, iteration,
                    mode='train',
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track,
                    if_update_timer=False,
                    train_data=None, train_batch_data=train_batch_data,
                    end_of_epoch=end_of_epoch,
                )

        return loss, output, labels


    def infer_bw_one_step(self, train_batch_data, device=None, args=None,
            epoch=None, iteration=None, end_of_epoch=False,
            tracker=None, metrics=None,
            move_to_gpu=True, model_train=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            hess_cal=False, hess_aprx=False, hess_tr_aprx=False,
            grad_per_sample=False,
            fim_tr=False, fim_tr_emp=False,
            parameters_crt_names=[],
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs
        ):
        """
            inference and BP without optimization
        """
        model = self.model

        if move_to_gpu:
            model.to(device)

        if model_train:
            model.train()
        else:
            model.eval()

        time_table = {}
        time_now = time.time()
        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)

        if clear_grad_bef_opt:
            self.optimizer.zero_grad()

        if self.args.model_out_feature:
            output, feat = model(x)
        else:
            output = model(x)
        loss = self.criterion(output, labels)
        loss_value = loss.item()
        time_table["FP"] = time.time() - time_now
        time_now = time.time()
        logging.debug(f" Whole model time FP: {time.time() - time_now}")

        loss.backward()

        if make_summary and (tracker is not None) and (metrics is not None):
            self.summarize(model, output, labels,
                    tracker, metrics,
                    loss_value,
                    epoch, iteration,
                    mode='train',
                    checkpoint_extra_name=checkpoint_extra_name,
                    things_to_track=things_to_track,
                    if_update_timer=False,
                    train_data=None, train_batch_data=train_batch_data,
                    end_of_epoch=end_of_epoch,
                )

        return loss, output, labels



    def test(self, test_data, device=None, args=None, epoch=None,
            tracker=None, metrics=None,
            move_to_gpu=True, make_summary=True,
            clear_grad_bef_opt=True, clear_grad_aft_opt=True,
            checkpoint_extra_name="centralized",
            things_to_track=[],
            **kwargs):

        model = self.model
        Acc_accm = 0.0

        model.eval()
        if move_to_gpu:
            model.to(device)
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data):
                x = x.to(device)
                labels = labels.to(device)
                real_batch_size = labels.shape[0]
                if self.args.model_input_channels == 3 and x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if self.args.model_out_feature:
                    output, feat = model(x)
                else:
                    output = model(x)

                loss = self.criterion(output, labels)
                if args.VHL and args.VHL_shift_test:
                    if args.VHL_label_style == "patch":
                        noise_label_shift = 0
                    elif args.VHL_label_style == "extra":
                        noise_label_shift = self.args.num_classes
                    else:
                        raise NotImplementedError

                    metric_stat = metrics.evaluate(loss, output, labels, pred_shift=noise_label_shift)
                    tracker.update_local_record(
                            'generator_track',
                            server_index=self.server_index,
                            client_index=self.client_index,
                            summary_n_samples=labels.shape[0]*1,
                            args=self.args,
                            PredShift_Loss=metric_stat["Loss"],
                            PredShift_Acc1=metric_stat["Acc1"],
                        )

                if make_summary and (tracker is not None) and (metrics is not None):
                    metric_stat = self.summarize(model, output, labels,
                            tracker, metrics,
                            loss.item(),
                            epoch, batch_idx,
                            mode='test',
                            checkpoint_extra_name=checkpoint_extra_name,
                            things_to_track=things_to_track,
                            if_update_timer=False,
                            train_data=test_data, train_batch_data=None,
                            end_of_epoch=False,
                        )
                    logging.debug(f"metric_stat[Acc1] is {metric_stat['Acc1']} ")
                    Acc_accm += metric_stat["Acc1"]
            logging.debug(f"Total is {Acc_accm} , averaged is {Acc_accm / (batch_idx+1)}")


    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None,
                        epoch=None, iteration=None, tracker=None, metrics=None):
        pass


    def feature_reduce(self, time_stamp=0, reduce_method="tSNE", extra_name="cent", postfix="",
        batch_data=None, data_loader=None, num_points=1000, save_checkpoint=True, save_image=False):
        # time_stamp represents epoch or round.
        data_tsne, labels = None, None
        if time_stamp in self.args.tSNE_track_epoch_list:
            if save_checkpoint:
                save_checkpoint_without_check(
                    self.args, self.save_checkpoints_config,
                    extra_name=extra_name,
                    epoch=time_stamp,
                    model_state_dict=self.get_model_params(),
                    optimizer_state_dict=None,
                    postfix=postfix,
                )
            if save_image:
                data_tsne, labels = self.dim_reducer.unsupervised_reduce(reduce_method=reduce_method, 
                    model=self.model, batch_data=batch_data, data_loader=data_loader, num_points=num_points)
                logging.info(f"data_tsne.shape: {data_tsne.shape}")
                if postfix is not None:
                    postfix_str = "-" + postfix
                else:
                    postfix_str = ""
                image_path = self.args.checkpoint_root_path + \
                    extra_name + setup_checkpoint_file_name_prefix(self.args) + \
                    "-epoch-"+str(time_stamp) + postfix_str +'.jpg'
                # save_image(tensor=x_cat, fp=image_path, nrow=8)
                # logging.info(f"data_tsne.deivce: {data_tsne.deivce}, labels.device: {labels.device} ")
                # logging.info(f"labels.device: {labels.device} ")

                plt.figure(figsize=(6, 4))
                plt.subplot(1, 2, 1)
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, alpha=0.6, 
                            cmap=plt.cm.get_cmap('rainbow', 10))
                plt.title("t-SNE")
                plt.savefig(image_path)
        return data_tsne, labels






