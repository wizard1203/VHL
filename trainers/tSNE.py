import logging

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import torch


def setup_tSNE_path(save_checkpoints_config,
        extra_name=None, epoch="init", postfix=None):

    postfix_str = "-" + postfix if postfix is not None else ""

    if extra_name is not None:
        save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
            + "tSNE-" + extra_name + "-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
            + "-epoch-"+str(epoch) + postfix_str + ".pdf"
    else:
        save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
            + "tSNE-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
            + "-epoch-"+str(epoch) + postfix_str + ".pdf"
    return save_activation_path


class Dim_Reducer(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def unsupervised_reduce(self, reduce_method="tSNE", 
        model=None, batch_data=None, data_loader=None, num_points=1000):
        if reduce_method == "tSNE":
            __reduce_method = TSNE(n_components=2, random_state=33)
            other_params = {}
        else:
            raise NotImplementedError

        if model is not None:
            data_input, labels = self.get_features(model=model, batch_data=batch_data,
                    data_loader=data_loader, num_points=num_points)
            data_tsne = __reduce_method.fit_transform(data_input, **other_params)
            labels = labels.numpy()
            model.to("cpu")
        else:
            if batch_data is not None:
                data_input, labels = batch_data
            else:
                raise NotImplementedError
            data_tsne = __reduce_method.fit_transform(data_input, **other_params)
            labels = labels.numpy()

        return data_tsne, labels


    def get_features(self, model=None, batch_data=None, data_loader=None, num_points=1000):
        if model is not None:
            model.eval()
            model = model.to(self.device)
            with torch.no_grad():
                if batch_data is not None:
                    data, labels = batch_data
                    data = data.to(self.device)
                    output, feat = model(data)
                    feat = feat.to('cpu')

                elif data_loader is not None:
                    feat_list = []
                    labels_list = []
                    loaded_num_points = 0
                    for i, batch_data in enumerate(data_loader):
                        data, labels = batch_data
                        data = data.to(self.device)
                        output, feat = model(data)
                        feat_list.append(feat)
                        labels_list.append(labels)
                        loaded_num_points += data.shape[0]
                        if num_points < loaded_num_points:
                            break
                    feat = torch.cat(feat_list, dim=0)[:num_points].to('cpu')
                    labels = torch.cat(labels_list, dim=0)[:num_points]
                else:
                    raise NotImplementedError
                logging.info(f"feat.shape: {feat.shape}")
                data_input = feat
                # data_tsne = __reduce_method.fit_transform(data_input, **other_params)
                model.to("cpu")
        else:
            raise NotImplementedError

        return data_input, labels

    def setup_tSNE_path(self, save_checkpoints_config,
            extra_name=None, epoch="init", postfix=None, file_format=".jpg"):

        postfix_str = "-" + postfix if postfix is not None else ""

        if extra_name is not None:
            save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
                + "tSNE-" + extra_name + "-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
                + "-epoch-"+str(epoch) + postfix_str + file_format
        else:
            save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
                + "tSNE-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
                + "-epoch-"+str(epoch) + postfix_str + file_format
        return save_activation_path

























