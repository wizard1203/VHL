from __future__ import print_function
import torch
import numpy as np
import time
import math
from scipy import stats

from . import utils


class NoneCompressor():
    def __init__(self):
        self.name = 'none'

    def compress(self, tensor):
        return tensor, tensor.dtype

    def decompress(self, tensor, ctc):
        z = tensor 
        return z 


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    def __init__(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 
        self.c = 0
        self.t = 0.
        self.name = 'topk'
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}


    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data):
        if name not in self.zero_conditions:
            self.zero_conditions[name] = torch.ones(data.numel(), dtype=torch.float32, device=data.device) 
        zero_condition = self.zero_conditions[name]
        zero_condition.fill_(1.0)
        zero_condition[self.indexes[name]] = 0.0
        self.zc = zero_condition

    def clear(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 


    # def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
    #     start = time.time()
    #     with torch.no_grad():
    #         if name not in self.residuals:
    #             self.residuals[name] = torch.zeros_like(tensor.data)
    #         # top-k solution
    #         numel = tensor.numel()
    #         k = max(int(numel * ratio), 1)
    #         self.current_ratio = ratio
    #         #tensor.data.add_(TopKCompressor.residuals[name].data)
    #         self._process_data_before_selecting(name, tensor.data)

    #         values, indexes = torch.topk(torch.abs(tensor.data), k=k)
    #         values = tensor.data[indexes]

    #         self.residuals[name].data = tensor.data + 0.0 
    #         self.residuals[name].data[indexes] = 0. 
    #         self.values[name] = values
    #         self.indexes[name] = indexes

    #         self._process_data_after_residual(name, tensor.data)

    #         return tensor, indexes, values

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.values[name] = values
            self.indexes[name] = indexes

            return tensor, indexes, values

    def decompress(self, tensor, original_tensor_size):
        return tensor


    def decompress_new(self, tensor, indexes, name=None, shape=None):
        '''
            Just decompress, without unflatten.
            Remember to do unflatter after decompress
        '''
        if shape is None:
            decompress_tensor = torch.zeros(
                self.shapes[name], dtype=tensor.dtype, device=tensor.device).view(-1)
            decompress_tensor[indexes] = tensor
            # decompress_tensor = torch.zeros(self.shapes[name]).view(-1)
            # decompress_tensor[indexes] = tensor.type(decompress_tensor.dtype)
            return decompress_tensor
        else:
            decompress_tensor = torch.zeros(
                self.shapes[name], dtype=tensor.dtype, device=tensor.device).view(-1)
            decompress_tensor[indexes] = tensor
            # decompress_tensor = torch.zeros(self.shapes[name]).view(-1)
            # decompress_tensor[indexes] = tensor.type(decompress_tensor.dtype)
            return decompress_tensor

    def flatten(self, tensor, name=None):
        ''' 
            flatten a tensor 
        '''
        self.shapes[name] = tensor.shape
        return tensor.view(-1)

    def unflatten(self, tensor, name=None, shape=None):
        ''' 
            unflatten a tensor 
        '''
        if shape is None:
            return tensor.view(self.shapes[name])
        else:
            return tensor.view(shape)

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape

    def get_residuals(self, name, like_tensor):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(like_tensor.data)
        return self.residuals[name]

    def add_residuals(self, included_indexes, name):
        with torch.no_grad():
            residuals = self.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = self.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[self.indexes[name]] += values.data
            #selected_indexes = TopKCompressor.indexes[name][indexes_t]
            #residuals.data[selected_indexes] = 0.0 
            #logger.info('residuals after: %f', torch.norm(TopKCompressor.residuals[name].data))




class TopKDDCompressor(TopKCompressor):
    """
    Error-feedback Top-k sparsification with delay decay
    """
    #ADD=0.3; MAX_DELAY=1.5 # For VGG-16: ADD=0.3; MAX_DELAY=1.5, @devidenorm1:acc@0.913300
    #ADD=0.6; MAX_DELAY=1.3 # For VGG-16: ADD=0.6; MAX_DELAY=1.3, @devidenorm2:acc@0.914900
    #ADD=1;MAX_DELAY=6 # For ResNet-20: Add=1; MAX_DELAY=6,@devidenorm
    #ADD=0.2;MAX_DELAY=1.3  # For ResNet-110: ADD=0.2; MAX_DELAY=1.3, @devidenorm1:acc@0.861500
    #zc = None

    def __init__(self):
        super().__init__()
        self.name = 'topkdd'
        self.ADD_MAXDELAY = (0.2, 1.3)

    def _process_data_before_selecting(self, name, data):
        data.mul_(self.delay_counters[name].data)

    def _process_data_after_residual(self, name, data):
        delay_counter = self.delay_counters[name]
        ADD, MAX_DELAY = self.ADD_MAXDELAY
        delay_counter.add_(ADD) 
        delay_counter[self.indexes[name]] = 1
        delay_counter[delay_counter>MAX_DELAY] = MAX_DELAY


class EFTopKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'eftopk'

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio
            #tensor.data.add_(TopKCompressor.residuals[name].data)
            self._process_data_before_selecting(name, tensor.data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0. 
            self.values[name] = values
            self.indexes[name] = indexes

            self._process_data_after_residual(name, tensor.data)

            return tensor, indexes, values

    def _process_data_before_selecting(self, name, data):
        data.add_(self.residuals[name].data)


class gTopKCompressor(TopKCompressor):
    def __init__(self):
        super().__init__()
        self.name = 'gtopk'


class gTopKEFCompressor(TopKCompressor):
    def __init__(self):
        super().__init__()
        self.name = 'gtopkef'

    def _process_data_before_selecting(self, name, data):
        data.add_(self.residuals[name].data)


class SequenceCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'sequence'
        self.iterations = {}
        self.sparsities = []

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        def _get_start_index(iteration, ratio, numel):
            k = int(numel * ratio)
            total_iters_per_round = int(math.ceil(float(numel) / k))
            start_index = iteration % total_iters_per_round
            start_index *= k
            if start_index + k > numel:
                k = numel - start_index
            return start_index, k
        with torch.no_grad():
            self.current_ratio = ratio
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            if name not in self.iterations:
                self.iterations[name] = 0
            tensor.add_(self.residuals[name].data)
            iteration = self.iterations[name]
            numel = tensor.numel()
            start_index, k = _get_start_index(iteration, ratio, numel)
            if name not in self.zero_conditions:
                self.zero_conditions[name] = torch.ones(numel, dtype=torch.float32, device=tensor.device) 
            zero_condition = self.zero_conditions[name]
            zero_condition.fill_(1.)
            zero_condition[start_index:start_index+k] = 0.
            self.zc = zero_condition

            self.residuals[name].data = tensor.data * zero_condition
            tensor.sub_(self.residuals[name].data)
            self.iterations[name] += 1
            return tensor, [start_index, k]

    def add_residuals(self, included_indexes, name):
        pass
            

class GaussianCompressor(TopKCompressor):
    """
    """

    def __init__(self):
        super().__init__()
        self.name = 'gaussian'
        self.iterations = {}
        self.sparsities = []

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            tensor.add_(self.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 5:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            indexes = indexes[0:k]
            values = tensor.data[indexes] 
            #print('gaussion vs topk: ', indexes.numel(), k)
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values


class RandomKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'randomk'
        self.counter = 0

    # def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
    #     with torch.no_grad():
    #         if name not in RandomKCompressor.residuals:
    #             self.residuals[name] = torch.zeros_like(tensor.data)
    #         numel = tensor.numel()
    #         k = max(int(numel * ratio), 1)
    #         self.current_ratio = ratio

    #         perm = torch.randperm(numel, device=tensor.device)
    #         self.counter += 1
    #         indexes = perm[:k]
    #         values = tensor.data[indexes] 
    #         self.residuals[name].data = tensor.data + 0.0 
    #         self.residuals[name].data[indexes] = 0.0

    #         self.values[name] = values
    #         self.indexes[name] = indexes
    #         self._process_data_after_residual(name, tensor)

    #         return tensor, indexes, values

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            perm = torch.randperm(numel, device=tensor.device)
            self.counter += 1
            indexes = perm[:k]
            values = tensor.data[indexes] 

            self.values[name] = values
            self.indexes[name] = indexes

            return tensor, indexes, values


class RandomKECCompressor(TopKCompressor):

    def __init__(self):
        super().__init__()
        self.name = 'randomkec'
        self.counter = 0


    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            tensor.add_(self.residuals[name].data)
            perm = torch.randperm(numel, device=tensor.device)
            self.counter += 1
            indexes = perm[:k]
            values = tensor.data[indexes] 
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values


class RandomKSameCompressor(TopKCompressor):

    def __init__(self):
        super().__init__()
        self.name = 'randomksame'
        self.counter = 0

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio
            #torch.manual_seed(RandomKSCompressor.counter)
            #if tensor.is_cuda:
            #    torch.cuda.manual_seed_all(RandomKSCompressor.counter)
            torch.manual_seed(self.counter)
            self.counter += 1
            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]
            values = tensor.data[indexes] 
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values


class RandomKSameECCompressor(TopKCompressor):

    def __init__(self):
        super().__init__()
        self.name = 'randomksameec'
        self.counter = 0

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio
            tensor.add_(self.residuals[name].data)
            #torch.manual_seed(RandomKSERCompressor.counter)
            #if tensor.is_cuda:
            #    torch.cuda.manual_seed_all(RandomKSERCompressor.counter)
            #RandomKSERCompressor.counter += 1
            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]
            values = tensor.data[indexes] 
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values


class DGCSamplingCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'dgcsampling' # Section 5 of the DGC paper, which samples 0.1% to 1% of the gradients to perform topk

    def _process_data_before_selecting(self, name, data):
        super()._process_data_before_selecting(name, data)
        data.add_(self.residuals[name].data)

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            self._process_data_before_selecting(name, tensor.data)

            abs_values = torch.abs(tensor.data)

            # First step sampling
            perm = torch.randperm(numel, device=tensor.device)
            if ratio >= 0.01:
                fk=k
            else:
                fk = int(numel * 0.01)
            sampled_indexes = perm[0:fk]
            sampled_values = abs_values[sampled_indexes]
            tmpvalues, tmpindexes = torch.topk(sampled_values, k=k)

            thres = tmpvalues[k-1]
            bool_indexes = abs_values > thres
            indexes = bool_indexes.nonzero().data.squeeze().view(-1)
            num_k = len(indexes)
            if num_k > 4*k/3:
                tmpvalues = abs_values[indexes] 
                values, tmpindexes = torch.topk(tmpvalues, k=k)
                indexes = indexes[tmpindexes]

            values = tensor.data[indexes] 
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values

class DGCSamplingDDCompressor(DGCSamplingCompressor):
    def __init__(self):
        super().__init__()
        self.name = 'dgcsamplingdd' # Section 5 of the DGC paper, which samples 0.1% to 1% of the gradients to perform topk
        self.delay_counters = {}
        #self.ADD_MAXDELAY = (0.2, 1.3) # default
        #self.ADD_MAXDELAY = (0.3, 1.8) # r1
        #self.ADD_MAXDELAY = (1, 6) # r2
        self.ADD_MAXDELAY = (1, 20) # r3

    def _process_data_before_selecting(self, name, data):
        if name not in self.delay_counters:
            self.delay_counters[name] = torch.ones_like(data)
        data.mul_(self.delay_counters[name].data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, data):
        super()._process_data_after_residual(name, data)
        delay_counter = self.delay_counters[name]
        ADD, MAX_DELAY = self.ADD_MAXDELAY
        delay_counter.add_(ADD) 
        delay_counter[self.indexes[name]] = 1
        delay_counter[delay_counter>MAX_DELAY] = MAX_DELAY


class TopKDDCompressor(TopKCompressor):
    """
    Error-feedback Top-k sparsification with delay decay
    """
    #ADD=0.3; MAX_DELAY=1.5 # For VGG-16: ADD=0.3; MAX_DELAY=1.5, @devidenorm1:acc@0.913300
    #ADD=0.6; MAX_DELAY=1.3 # For VGG-16: ADD=0.6; MAX_DELAY=1.3, @devidenorm2:acc@0.914900
    #ADD=1;MAX_DELAY=6 # For ResNet-20: Add=1; MAX_DELAY=6,@devidenorm
    #ADD=0.2;MAX_DELAY=1.3  # For ResNet-110: ADD=0.2; MAX_DELAY=1.3, @devidenorm1:acc@0.861500
    #zc = None

    def __init__(self):
        super(self.__class__, self).__init__()
        self.name = 'topkdd'
        self.delay_counters = {}
        self.ADD_MAXDELAY = (0.2, 1.3)

    def _process_data_before_selecting(self, name, data):
        super()._process_data_before_selecting(name, data)
        if name not in self.delay_counters:
            self.delay_counters[name] = torch.ones_like(data)
        data.mul_(self.delay_counters[name].data)

    def _process_data_after_residual(self, name, data):
        super()._process_data_after_residual(name, data)
        delay_counter = self.delay_counters[name]
        ADD, MAX_DELAY = self.ADD_MAXDELAY
        delay_counter.add_(ADD) 
        delay_counter[self.indexes[name]] = 1
        delay_counter[delay_counter>MAX_DELAY] = MAX_DELAY


class EFTopKDDCompressor(TopKCompressor):
    """
    Error-feedback Top-k sparsification with delay decay
    """
    #ADD=0.3; MAX_DELAY=1.5 # For VGG-16: ADD=0.3; MAX_DELAY=1.5, @devidenorm1:acc@0.913300
    #ADD=0.6; MAX_DELAY=1.3 # For VGG-16: ADD=0.6; MAX_DELAY=1.3, @devidenorm2:acc@0.914900
    #ADD=1;MAX_DELAY=6 # For ResNet-20: Add=1; MAX_DELAY=6,@devidenorm
    #ADD=0.2;MAX_DELAY=1.3  # For ResNet-110: ADD=0.2; MAX_DELAY=1.3, @devidenorm1:acc@0.861500

    def __init__(self):
        #super(self.__class__, self).__init__()
        super().__init__()
        self.name = 'eftopkdd'
        self.delay_counters = {}
        self.ADD_MAXDELAY = (0.2, 1.3) #
        #self.ADD_MAXDELAY = (1, 6) # r1
        #self.ADD_MAXDELAY = (1, 20) # r3
        #self.ADD_MAXDELAY = (1, 200) # r4
        #self.ADD_MAXDELAY = (0.1, 2.0) # r5

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio
            #tensor.data.add_(TopKCompressor.residuals[name].data)
            self._process_data_before_selecting(name, tensor.data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0. 
            self.values[name] = values
            self.indexes[name] = indexes

            self._process_data_after_residual(name, tensor.data)

            return tensor, indexes, values

    def _process_data_before_selecting(self, name, data):
        if name not in self.delay_counters:
            self.delay_counters[name] = torch.ones_like(data)
        data.mul_(self.delay_counters[name].data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, data):
        super()._process_data_after_residual(name, data)
        delay_counter = self.delay_counters[name]
        ADD, MAX_DELAY = self.ADD_MAXDELAY
        delay_counter.add_(ADD) 
        delay_counter[self.indexes[name]] = 1
        delay_counter[delay_counter>MAX_DELAY] = MAX_DELAY

class EFTopKDDCompressorR1(EFTopKDDCompressor):
    def __init__(self):
        super().__init__()
        self.name = 'eftopkddr1'
        self.ADD_MAXDELAY = (1, 6) 

class EFTopKDDCompressorR2(EFTopKDDCompressor):
    def __init__(self):
        super().__init__()
        self.name = 'eftopkddr2'
        self.ADD_MAXDELAY = (1, 20) 

class EFTopKDDCompressorR3(EFTopKDDCompressor):
    def __init__(self):
        super().__init__()
        self.name = 'eftopkddr3'
        self.ADD_MAXDELAY = (0.1, 2.0)

class EFTopKDDCompressorR4(EFTopKDDCompressor):
    def __init__(self):
        super().__init__()
        self.name = 'eftopkddr4'
        self.ADD_MAXDELAY = (2, 40)



class EFTopKLookForwardCompressor(TopKCompressor):
    """
    Error-feedback Top-k sparsification with look forward 
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self.name = 'eftopklookforward'
        self.delay_counters = {}
        #self.ADD_MAXDELAY = (0.2, 1.3) #
        self.ADD_MAXDELAY = (1, 6) # r1
        #self.ADD_MAXDELAY = (1, 20) # r3
        #self.ADD_MAXDELAY = (1, 200) # r4
        #self.ADD_MAXDELAY = (0.1, 2.0) # r5

    def _process_data_before_selecting(self, name, data):
        if name not in self.delay_counters:
            self.delay_counters[name] = torch.ones_like(data)
        data.mul_(self.delay_counters[name].data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, data):
        super()._process_data_after_residual(name, data)
        delay_counter = self.delay_counters[name]
        ADD, MAX_DELAY = self.ADD_MAXDELAY
        delay_counter.add_(ADD) 
        delay_counter[self.indexes[name]] = 1.0
        data[self.indexes[name]] = 0.0
        delay_counter[delay_counter>MAX_DELAY] = MAX_DELAY


class EFTopKDecayCompressor(TopKCompressor):
    """
    Error-feedback Top-k sparsification with decay
    """
    def __init__(self):
        super(self.__class__, self).__init__()
        self.name = 'eftopkdecay'
        self.delay_counters = {}
        self.ADD_MAXDELAY = (0.02, 0.5)

    def _process_data_before_selecting(self, name, data):
        if name not in self.delay_counters:
            self.delay_counters[name] = torch.ones_like(data)
        data.mul_(self.delay_counters[name].data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, data):
        super()._process_data_after_residual(name, data)
        delay_counter = self.delay_counters[name]
        ADD, MAX_DELAY = self.ADD_MAXDELAY
        #x_norm = float(data.norm())
        #topk_norm = float(self.values[name].norm())
        #ratio = (2 * x_norm * topk_norm - topk_norm*topk_norm) / (x_norm * x_norm)
        #delta = (2.0 * x_norm * topk_norm - topk_norm*topk_norm) / (x_norm * x_norm)
        #ratio = torch.sqrt((delta*delta*delta - delta*delta + 2) / delta) / (1-delta+1e-6)
        ratio = 0. #1e-3 #delta/(1.0-delta) 
        delay_counter.fill_(1.0) 
        delay_counter[self.indexes[name]] = ratio #self.current_ratio


class AdaSamplingCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'adasampling' 

    def _process_data_before_selecting(self, name, data):
        super()._process_data_before_selecting(name, data)
        data.add_(self.residuals[name].data)

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            self._process_data_before_selecting(name, tensor.data)

            abs_values = torch.abs(tensor.data)
            values, indexes = torch.topk(abs_values, k=k)
            values = tensor.data[indexes]

            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values



class QuantizationCompressor(object):
    def __init__(self):
        self.name = 'quant'
        self.residuals = {}
        self.values = {} 
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}

    def get_naive_quantize(self, x, s, is_biased=False):
        norm = x.norm(p=2)
        # calculate the quantization value of tensor `x` at level `log_2 s`.
        level_float = s * x.abs() / norm
        # floor to quantization
        previous_level = torch.floor(level_float)
        return torch.sign(x) * norm * previous_level / s

    def compress(self, tensor, name=None, quantize_level=32, is_biased=True):
        if quantize_level != 32:
            s = 2 ** quantize_level - 1
            values = self.get_naive_quantize(tensor, s, is_biased)
        else:
            values = tensor
        return values

    def decompress_new(self, tensor):
        return tensor

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape




class QSGDCompressor(object):
    def __init__(self):
        self.name = 'qsgd'
        self.residuals = {}
        self.values = {} 
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}

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

    def compress(self, tensor, name=None, quantize_level=32, is_biased=True):
        if quantize_level != 32:
            s = 2 ** quantize_level - 1
            values = self.get_qsgd(tensor, s, is_biased)
        else:
            values = tensor
        return values

    def decompress_new(self, tensor):
        return tensor

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape





#import bit2byte
class SignCompressor(object):
    """Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote"""
    def __init__(self):
        self.zc = None
        self.name = 'signum'

    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data, original_tensor):
        pass

    def packing(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        packed_data = src_tensor
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32, -1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        return dst_tensor, packed_data

    def unpacking(self, src_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(
            src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32
        )
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = -new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        return new_tensor

    def majority_vote(self, src_tensor_list):
        voter_num = len(src_tensor_list)
        src_tensor = torch.stack(src_tensor_list)
        src_tensor = src_tensor.view(-1)
        full_size = 32 * len(src_tensor)
        new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = -new_tensor.add_(-1)
        # sum
        new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
        new_tensor = torch.sum(new_tensor, 0)
        new_tensor = new_tensor.view(-1, 32).permute(1, 0)
        new_tensor = torch.sign(new_tensor)
        new_tensor = bit2byte.packing(new_tensor)
        new_tensor = new_tensor.to(dtype=torch.int32)
        return new_tensor

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num

    #def compress(self, src_tensor):
    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        self._process_data_before_selecting(name, tensor)
        packed_tensor, packed_data = self.packing(tensor)
        self._process_data_after_residual(name, packed_data, tensor)
        return packed_tensor, None, None

    def decompress(self, tensor, original_tensor_size):
        dst_tensor = self.unpacking(tensor, original_tensor_size)
        return dst_tensor

class EFSignCompressor(SignCompressor):
    def __init__(self):
        super().__init__()
        self.zc = None
        self.name = 'efsignum'
        self.residuals = {}

    def _process_data_before_selecting(self, name, data):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, packed_data, original_tensor):
        self.residuals[name] = original_tensor - packed_data

class EFDecaySignCompressor(EFSignCompressor):
    def __init__(self):
        super().__init__()
        self.zc = None
        self.name = 'efsignumdecay'
        self.residuals = {}
        self.delay_counters = {}

    def _process_data_before_selecting(self, name, data):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(data)
        if name not in self.delay_counters:
            self.delay_counters[name] = torch.ones_like(data)
        data.mul_(self.delay_counters[name].data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, packed_data, original_tensor):
        delay_counter = self.delay_counters[name]
        res = original_tensor - packed_data
        self.residuals[name] = res 
        error = res / (original_tensor+1e-6)
        indexes = error.abs() < 1e-3
        nonzero = indexes.nonzero().data.squeeze().numel()
        ratio = float(nonzero) / indexes.numel()
        delay_counter.fill_(1.0+ratio)
        delay_counter[indexes] = ratio 




compressors = {
        'no': NoneCompressor,
        None: NoneCompressor,
        'topk': TopKCompressor,
        'eftopk': EFTopKCompressor, #TopK with error-feedback
        'gtopk': gTopKCompressor, #gTopK without error-feedback
        'gtopkef': gTopKEFCompressor, #gTopK with error-feedback
        'sequence': SequenceCompressor, #Sequence indices with error-feedback
        'gaussian': GaussianCompressor, #GaussianK with error-feedback
        'randomk': RandomKCompressor, #RandomK without error-feedback
        'randomkec': RandomKECCompressor, #RandomK with error-feedback
        'randomksame': RandomKSameCompressor, #RandomK with the same indices without error-feedback
        'randomksameec': RandomKSameECCompressor, #RandomK with the same indices with error-feedback
        'dgcsampling': DGCSamplingCompressor, #DGC (doubling sampling) with error-feedback

        'topkdd': TopKDDCompressor, #TopK without error-feedback and delay decay
        'eftopkdd': EFTopKDDCompressor, #TopK with error-feedback and delay decay
        'eftopkddr1': EFTopKDDCompressorR1, 
        'eftopkddr2': EFTopKDDCompressorR2, 
        'eftopkddr3': EFTopKDDCompressorR3, 
        'eftopkddr4': EFTopKDDCompressorR4, 
        'eftopkdecay': EFTopKDecayCompressor, #TopK with error-feedback and decay
        'eftopklookforward': EFTopKLookForwardCompressor, #TopK with error-feedback and look forward 
        'dgcsamplingdd': DGCSamplingDDCompressor, #DGC with error-feedback and delay decay

        'quantize': QuantizationCompressor, # Naive Quantization Compressor
        'qsgd': QSGDCompressor, # QSGD Quantization Compressor

        'sign': SignCompressor,
        'efsign': EFSignCompressor,
        'efsigndecay': EFDecaySignCompressor,
        }


def test_gaussion_thres():
    set_mean = 0.0; set_std = 0.5
    d = np.random.normal(set_mean, set_std, 10000)
    k2, p = stats.normaltest(d)
    print(p)
    nnz = np.count_nonzero(d)
    mean = np.mean(d)
    std = np.std(d)
    print('size:%d, nnz: %d' % (d.size, nnz))
    print(set_mean, set_std)
    print(mean, std)
    copyd = d.copy()
    thres = 3*std
    d[np.abs(d) < thres] = 0
    pvalue = 1-np.count_nonzero(d)*1.0/d.size
    print('size:%d, p-value: %f' % (d.size, pvalue))
    left_thres, right_thres = utils.gen_threshold_from_normal_distribution(pvalue, mean, std)
    print('real thres:%f, gen thres: %f' % (thres, right_thres))


if __name__ == '__main__':
    #test_gaussion_thres()
    compressor_str = 'signum'
    compressor = compressors[compressor_str]()
    z = torch.rand(128, 256)
    compressed_tensor, _, _ = compressor.compress(z)
    print('compressed shape: ', compressed_tensor.shape)
    decompressed_tensor = compressor.decompress(compressed_tensor, z.size())
    print('decompressed shape: ', decompressed_tensor.shape)
    diff = (decompressed_tensor - z).norm()
    print('difff norm: ', diff)

