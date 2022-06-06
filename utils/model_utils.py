from collections.abc import Iterable

import logging

from .data_utils import scan_model_with_depth


def build_param_groups(named_parameters, named_modules, layers_depth):
    param_groups = []

    for name, param in named_parameters.items():
        module_name = '.'.join(name.split('.')[:-1])
        logging.info(f"In build_param_groups, module_name: {module_name} ")
        if module_name in named_modules:
            pass
            param_groups.append({'params': param, "param_name": name, "layer_name": module_name, "depth": layers_depth[module_name]})
        else:
            logging.info(f"In build_param_groups, name:{name}, module_name: {module_name} ")
            # raise NotImplementedError
        # param_groups.append({'params': model.classifier.parameters(), 'lr': 1e-3})

    return param_groups


def build_layer_params(named_parameters, named_modules, param_types=["Conv2d","Linear","BatchNorm2d"]):
    # param_groups = []
    layer_params = {}
    layers_depth = {}
    current_depth = 0

    for name, param in named_parameters.items():
        module_name = '.'.join(name.split('.')[:-1])
        if (len(param_types) > 0 and type(named_modules[module_name]).__name__ in param_types) or \
            len(param_types) == 0:
            if name.split('.')[-1] == "num_batches_tracked":
                # logging.info(f"In build_layer_params, NOT add module_name: {module_name} param name:{name} ")
                continue
            else:
                # logging.info(f"In build_layer_params, add module_name: {module_name} param name:{name} ")
                pass
            if module_name not in layer_params:
                layer_params[module_name] = []
                current_depth += 1
            if module_name not in layers_depth:
                layers_depth[module_name] = current_depth
            layer_params[module_name].append({'params': param, "param_name": name, "layer_name": module_name, "depth": layers_depth[module_name]})
        else:
            # pass
            logging.info(f"In build_layer_params, NOT add name:{name}, module_name: {module_name} ")
            # raise NotImplementedError
        # param_groups.append({'params': model.classifier.parameters(), 'lr': 1e-3})

    return layer_params



def scan_model_with_depth(model, param_types=[]):
    
    named_parameters = dict(model.named_parameters())
    named_modules = dict(model.named_modules())

    layers_depth = {}
    current_depth = 0

    for name, param in named_parameters.items():
        module_name = '.'.join(name.split('.')[:-1])
        if (len(param_types) > 0 and type(named_modules[module_name]).__name__ in param_types) or \
            len(param_types) == 0:
            if module_name not in layers_depth:
                current_depth += 1
                layers_depth[module_name] = current_depth
        else:
            pass
        # params_group.append({'params': model.classifier.parameters(), 'lr': 1e-3})
        # params_group.append({'params': param, "layer_name": "module_name", "depth": layers_depth[module_name]})

    return named_parameters, named_modules, layers_depth



# def build_param_groups(model_dict, named_modules, layers_depth):
#     param_groups = []

#     for name, param in model_dict.items():
#         module_name = '.'.join(name.split('.')[:-1])
#         assert module_name in named_modules
#         logging.info(f"In build_param_groups, module_name: {module_name} ")
#         # param_groups.append({'params': model.classifier.parameters(), 'lr': 1e-3})
#         param_groups.append({'params': param, "param_name": name, "layer_name": module_name, "depth": layers_depth[module_name]})

#     return param_groups








def get_actual_layer_names(model, layer_alias):
    layer_names = []
    name_alias_map = {}
    for alias in layer_alias:
        layer_name = model.layers_name_map[alias]
        layer_names.append(layer_name)
        name_alias_map[layer_name] = alias
    return layer_names, name_alias_map


def set_freeze_by_names(model, layer_alias=None, layer_names=None, freeze=True):
    # if not isinstance(layer_names, Iterable):
    #     layer_names = [layer_names]

    if layer_names is None and layer_alias is not None:
        layer_names, name_alias_map = get_actual_layer_names(model, layer_alias)
    elif layer_names is None and layer_alias is None:
        logging.info(f"No layer_names ")
        raise NotImplementedError
    else:
        pass

    logging.info(f"layer_names: {layer_names}")
    # for name, child in model.named_children():
    for name, module in model.named_modules():
        if name not in layer_names:
            continue
        else:
            logging.info(f"module: {module} is {'freezed' if freeze else 'NOT freezed'}")
            for param in module.parameters():
                param.requires_grad = not freeze

def freeze_by_names(model, layer_alias=None, layer_names=None):
    set_freeze_by_names(model, layer_alias=layer_alias, layer_names=layer_names, freeze=True)

def unfreeze_by_names(model, layer_alias=None, layer_names=None):
    set_freeze_by_names(model, layer_alias=layer_alias, layer_names=layer_names, freeze=False)



def set_freeze_by_depth(named_parameters, named_modules, layers_depth,
                layers_freeze_list=[], freeze=True):

    logging.info(f"layer_names: {layer_names}")

    # for name, param in named_parameters.items():
    #     module_name = '.'.join(name.split('.')[:-1])
        # if type(named_modules[module_name]).__name__ in param_types:
        #     filtered_parameters_crt_names.append(name)

    logging.info(f"module: {named_modules[module_name]} is {'freezed' if freeze else 'NOT freezed'}")
    # for param in module.parameters():
    for name, param in named_parameters.items():
        module_name = '.'.join(name.split('.')[:-1])
        assert module_name in named_modules
        if layers_depth[module_name] in layers_freeze_list:
            param.requires_grad = not freeze


def freeze_by_depth(named_parameters, named_modules, layers_depth,
                layers_freeze_list=[]):
    set_freeze_by_depth(named_parameters, named_modules, layers_depth,
                layers_freeze_list, freeze=True)

def unfreeze_by_depth(named_parameters, named_modules, layers_depth,
                layers_freeze_list=[]):
    set_freeze_by_depth(named_parameters, named_modules, layers_depth,
                layers_freeze_list=[], freeze=False)






def get_modules_by_names(model, layer_alias=None, layer_names=None):
    if layer_names is None and layer_alias is not None:
        layer_names, name_alias_map = get_actual_layer_names(model, layer_alias)
    elif layer_names is None and layer_alias is None:
        logging.info(f"No layer_names ")
        raise NotImplementedError
    else:
        pass

    module_dict = {}
    logging.info(f"layer_names: {layer_names}")
    for name, module in model.named_modules():
        if name not in layer_names:
            continue
        else:
            module_dict[name_alias_map[name]] = module
            logging.info(f"Add module: {module} into module_dict")
    return module_dict


# def set_freeze_by_idxs(model, idxs, freeze=True):
#     if not isinstance(idxs, Iterable):
#         idxs = [idxs]
#     # num_child = len(list(model.children()))
#     # idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
#     # for idx, child in enumerate(model.children()):
#     #     if idx not in idxs:
#     #         continue
#     #     for param in child.parameters():
#     #         param.requires_grad = not freeze
#     num_params = len(list(model.named_parameters()))
#     idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
#     for idx, child in enumerate(model.children()):
#         if idx not in idxs:
#             continue
#         for param in child.parameters():
#             param.requires_grad = not freeze

# def freeze_by_idxs(model, idxs):
#     set_freeze_by_idxs(model, idxs, True)

# def unfreeze_by_idxs(model, idxs):
#     set_freeze_by_idxs(model, idxs, False)












