import math
import logging

import numpy as np
import torch


# Not easy to install this lib
# import bit2byte


def get_n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


"""define some general compressors, e.g., topk, randomk, sign"""


class SparsificationCompressor(object):
    def get_top_k(self, x, ratio):
        """it will sample the top 1-ratio of the samples."""
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # get indices and the corresponding values
        if top_k == 1:
            _, selected_indices = torch.max(x_data.abs(), dim=0, keepdim=True)
        else:
            _, selected_indices = torch.topk(
                x_data.abs(), top_k, largest=True, sorted=False
            )
        return x_data[selected_indices], selected_indices

    def get_mask(self, flatten_arr, indices):
        mask = torch.zeros_like(flatten_arr)
        mask[indices] = 1

        mask = mask.byte()
        return mask.float(), (~mask).float()

    def get_random_k(self, x, ratio, is_biased=True):
        """it will randomly sample the 1-ratio of the samples."""
        # get tensor size.
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # random sample the k indices.
        selected_indices = np.random.choice(x_len, top_k, replace=False)
        selected_indices = torch.LongTensor(selected_indices).to(x.device)

        if is_biased:
            return x_data[selected_indices], selected_indices
        else:
            return x_len / top_k * x_data[selected_indices], selected_indices

    def compress(self, arr, op, compress_ratio, is_biased):
        if "topk" in op:
            values, indices = self.get_top_k(arr, compress_ratio)
        elif "randomk" in op:
            values, indices = self.get_random_k(arr, compress_ratio)
        else:
            raise NotImplementedError

        # n_bits = get_n_bits(values) + get_n_bits(indices)
        return values, indices

    def uncompress(self, values, indices, selected_shapes, original_shapes):
        """
            values:             compressed values of multiple tensors
            indices:            indices of selected values in their original tensors 
                                (means that indices are not unique)
            selected_shapes:    list of sizes of each compressed tensor
            original_shapes:    list of sizes of each original tensor
        """
        # apply each param.
        sync_pointer = 0
        pointer = 0

        _q_values, _q_indices = [], []
        # logging.debug("values %s" % values)
        # logging.debug("values[0:5] %s" % values[0:5])
        # logging.info("indices: {}, indices.dtype:{}".format(
        # indices, indices.dtype))
        # if (indices == 51034000).any():
        # raise RuntimeError
        # logging.debug("indices %s" % indices)
        # logging.debug("selected_shapes %s" % selected_shapes)
        # logging.debug("original_shapes %s" % original_shapes)
        for idx, n_sparse_value in enumerate(selected_shapes):
            # get value and indice for the current param.
            # logging.debug("n_sparse_value: %s" % n_sparse_value)
            _q_value = values[sync_pointer : sync_pointer + n_sparse_value]
            _q_indice = pointer + indices[sync_pointer : sync_pointer + n_sparse_value]
            _q_values += [_q_value]
            _q_indices += [_q_indice]
            # logging.info("*********************pointer: {}".format(pointer))
            # logging.info("*********************indices[sync_pointer : sync_pointer + n_sparse_value]: {}".format(
            # indices[sync_pointer : sync_pointer + n_sparse_value]))
            # update the pointers.
            sync_pointer += n_sparse_value
            pointer += original_shapes[idx][1]
            # logging.info("*********************pointer: {}".format(pointer))
            # logging.info("_q_indice: {}, _q_indice.dtype:{}".format(_q_indice, _q_indice.dtype))
            # if (_q_indice == 51034000).any():
            #     raise RuntimeError
        return torch.cat(_q_values), torch.cat(_q_indices).long()


class QuantizationCompressor(object):
    def get_qsgd(self, x, s, is_biased=False):
        norm = x.norm(p=2)
        # calculate the quantization value of tensor `x` at level `log_2 s`.
        level_float = s * x.abs() / norm
        # floor to quantization
        previous_level = torch.floor(level_float)
        # add the stochastic quantization, to preserve the value in expectation.
        is_next_level = (torch.rand_like(x) < (level_float - previous_level)).float()
        new_level = previous_level + is_next_level

        scale = 1
        if is_biased:
            d = x.nelement()
            # Variance bound of QSGD
            scale = 1.0 / (min(d / (s ** 2), math.sqrt(d) / s) + 1.0)
        return scale * torch.sign(x) * norm * new_level / s

    def get_naive_quantize(self, x, s, is_biased=False):
        norm = x.norm(p=2)
        # calculate the quantization value of tensor `x` at level `log_2 s`.
        level_float = s * x.abs() / norm
        # floor to quantization
        previous_level = torch.floor(level_float)
        return torch.sign(x) * norm * previous_level / s


    def qsgd_quantize_numpy(self, x, s, is_biased=False):
        """quantize the tensor x in d level on the absolute value coef wise"""
        norm = np.sqrt(np.sum(np.square(x)))
        # calculate the quantization value of tensor `x` at level `log_2 s`.
        level_float = s * np.abs(x) / norm
        # floor to quantization
        previous_level = np.floor(level_float)
        # add the stochastic quantization, to preserve the value in expectation.
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level

        scale = 1
        if is_biased:
            d = len(x)
            # Variance bound of QSGD
            scale = 1.0 / (np.minimum(d / s ** 2, np.sqrt(d) / s) + 1.0)
        return scale * np.sign(x) * norm * new_level / s

    def compress(self, arr, op, quantize_level, is_biased):
        if quantize_level != 32:
            s = 2 ** quantize_level - 1
            if op == "quant":
                values = self.get_naive_quantize(arr, s, is_biased)
            elif op == "qsgd":
                values = self.get_qsgd(arr, s, is_biased)
            else:
                raise NotImplementedError
        else:
            values = arr
        return values

    def uncompress(self, arr):
        return arr


# class SignCompressor(object):
#     """Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote"""

#     def packing(self, src_tensor):
#         src_tensor = torch.sign(src_tensor)
#         src_tensor_size = src_tensor.size()
#         src_tensor = src_tensor.view(-1)
#         src_len = len(src_tensor)
#         add_elm = 32 - (src_len % 32)
#         if src_len % 32 == 0:
#             add_elm = 0
#         new_tensor = torch.zeros(
#             [add_elm], dtype=torch.float32, device=src_tensor.device
#         )
#         src_tensor = torch.cat((src_tensor, new_tensor), 0)
#         src_tensor = src_tensor.view(32, -1)
#         src_tensor = src_tensor.to(dtype=torch.int32)
#         dst_tensor = bit2byte.packing(src_tensor)
#         dst_tensor = dst_tensor.to(dtype=torch.int32)
#         return dst_tensor, src_tensor_size

#     def unpacking(self, src_tensor, src_tensor_size):
#         src_element_num = self.element_num(src_tensor_size)
#         add_elm = 32 - (src_element_num % 32)
#         if src_element_num % 32 == 0:
#             add_elm = 0
#         src_tensor = src_tensor.int()
#         new_tensor = torch.ones(
#             src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32
#         )
#         new_tensor = new_tensor.view(32, -1)
#         new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
#         new_tensor = new_tensor.view(-1)
#         new_tensor = new_tensor[:src_element_num]
#         new_tensor = new_tensor.view(src_tensor_size)
#         new_tensor = -new_tensor.add_(-1)
#         new_tensor = new_tensor.float()
#         return new_tensor

#     def majority_vote(self, src_tensor_list):
#         voter_num = len(src_tensor_list)
#         src_tensor = torch.stack(src_tensor_list)
#         src_tensor = src_tensor.view(-1)
#         full_size = 32 * len(src_tensor)
#         new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
#         new_tensor = new_tensor.view(32, -1)
#         new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
#         new_tensor = -new_tensor.add_(-1)
#         # sum
#         new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
#         new_tensor = torch.sum(new_tensor, 0)
#         new_tensor = new_tensor.view(-1, 32).permute(1, 0)
#         new_tensor = torch.sign(new_tensor)
#         new_tensor = bit2byte.packing(new_tensor)
#         new_tensor = new_tensor.to(dtype=torch.int32)
#         return new_tensor

#     def element_num(self, size):
#         num = 1
#         for i in range(len(size)):
#             num *= size[i]
#         return num

#     def compress(self, src_tensor):
#         return self.packing(src_tensor)

#     def uncompress(self, src_tensor, src_tensor_size):
#         dst_tensor = self.unpacking(src_tensor, src_tensor_size)
#         return dst_tensor
