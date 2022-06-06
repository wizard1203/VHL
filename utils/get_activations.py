import logging 
import h5py
import torch
import numpy as np

def setup_save_activation_path(save_checkpoints_config,
        extra_name=None, epoch="init", postfix=None):

    postfix_str = "-" + postfix if postfix is not None else ""

    if extra_name is not None:
        save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
            + "activations-" + extra_name + "-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
            + "-epoch-"+str(epoch) + postfix_str + ".h5"
    else:
        save_activation_path = save_checkpoints_config["checkpoint_root_path"] \
            + "activations-" + save_checkpoints_config["checkpoint_file_name_prefix"] \
            + "-epoch-"+str(epoch) + postfix_str + ".h5"
    return save_activation_path


def register_get_activation_hooks(model, layers_list):

    activation = {}
    hook_handle_dict = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for layer_name in layers_list:
        tree_layer_names = layer_name.split(".")
        layer = model
        for single_layer_name in tree_layer_names:
            if single_layer_name.isdigit():
                layer = layer[int(single_layer_name)]
            else:
                layer = getattr(layer, single_layer_name)
        hook_handle = layer.register_forward_hook(get_activation(layer_name))
        hook_handle_dict[layer] = hook_handle
    return activation, hook_handle_dict

def get_dataset_activation(trainloader, model, activation, device, save_activation_path=None):
    activation_dataset = {}
    with torch.no_grad():
        for i, data in enumerate(trainloader):
            # if i == max_iter:
            #     break
            inputs, labels = data
            # inputs = inputs.cuda()
            inputs = inputs.to(device)
            _ = model(inputs)
            for layer, value in activation.items():
                if layer not in activation_dataset:
                    activation_dataset[layer] = value
                else:
                    activation_dataset[layer] = torch.cat((activation_dataset[layer], value), 0)
            logging.info('Got %f th iter activations...' %(i))

        for layer, value in activation_dataset.items():
            logging.info('activations {} shape: {}'.format(layer, value.shape))

    if save_activation_path is not None:
        act_file = h5py.File(save_activation_path, 'w')
        for layer, act in activation_dataset.items():
            act_file[layer] = act.cpu()
        act_file.close()
    return activation_dataset



def load_dataset_activation(save_activation_path, layers_list=None):
    act_file = h5py.File(save_activation_path, 'r')
    act = {}
    logging.info(act_file)
    for layer, _ in act_file.items():
        if layers_list is not None:
            if layer not in layers_list:
                logging.info("Skip layer: {}, not in input layer list {}".format(
                    layer, layers_list))
                continue
        act[layer] = torch.as_tensor(np.array(act_file[layer]))
        logging.info("loaded layer: {}, data shape: {}".format(layer, act_file[layer].shape) )

    if layers_list is not None:
        for layer in layers_list:
            if layer not in list(act_file.keys()):
                logging.warning()("Not found layer: {}, in {}".format(layer, save_activation_path) )


    act_file.close()
    return act



def save_activation(save_activation_path, activation_dataset):
    act_file = h5py.File(save_activation_path, 'w')
    for layer, act in activation_dataset.items():
        act_file[layer] = act
    act_file.close()








if __name__ == "__main__":
    pass













