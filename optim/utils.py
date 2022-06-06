import hashlib
import time
import os
import numpy as np
import scipy.stats as stats
from collections.abc import Iterable


# def freeze_layer(args, optimizer, model, layer_index=None, layer_name=None):
#     pass




# def unfreeze_layer(args, optimizer, model, layer_index=None, layer_name=None):
#     pass






def gen_random_id():
    id_ = hashlib.sha256()
    id_.update(str(time.time()))
    return id_.hexdigest()

def create_path(relative_path):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relative_path)
    if not os.path.isdir(filename):
        try:
            #os.mkdir(filename)
            os.makedirs(filename)
        except:
            pass

def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def autolabel(rects, ax, label, rotation=90):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_y() + rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            label,
            ha='center', va='bottom', rotation=rotation)

def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:]
    return indexes, tensor[indexes]

def get_approximate_sigma_scale(density):
    sigma_scale = 1
    if density > 0.7:
        sigma_scale = 0.5
    elif density <= 0.7 and density > 0.05:
        sigma_scale = 1.5
    elif density <= 0.05 and density > 0.01:
        sigma_scale = 2.0
    else:
        sigma_scale = 3.0
    return sigma_scale



def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)


s=2.18896957e-10 #P102-100
#s=4.99671953e-10 #V100
#a=0.002661810655986525 # small message <1M
#b=1.3644874178760432e-08 # small message <1M
GbE_multi_p_ab_small = {
        2: (1.6e-3, 1.0e-8),
        4: (2.7e-3, 1.3e-8),
        8: (4.0e-3, 1.5e-8),
        #16: (1.1e-2, 1.7e-8)
        16: (1.7e-3, 1.7e-8) #  ImageNet
        #16: (0.05e-2, 0.28e-8) # Inceptionv4 8 layers
        }


GbE_multi_p_ab_large = {
        2: (4.4e-3, 5.8e-9),
        4: (5.6e-3, 7.4e-9),
        8: (7.68e-3, 8.2e-9),
        16: (2.1e-3, 1.7e-8) # good for imagenet
        }

tenGbE_multi_p_ab = {
        2: (1.5e-5, 5.7e-11),
        4: (3.6e-5, 1.1e-10),
        8: (8.5e-5, 1.4e-10),
        16: (1.4e-4, 2.0e-10)
        }



#a=0.015890215705869848 # large message >1M
#b=8.594593687256138e-09 # large message >1M

def topk_perf_model(x, s=s):
    """
    x is the number of parameters
    Return: s * x * log2(x)
    """
    if x == 0.0:
        return 0.0
    return s * x * np.log2(x)

def allgather_perf_model(x, P, density=0.001, eth='GbE'):
    """
    x is the number of parameters
    Return: t = a + b * x
    """
    if x == 0:
        return 0.0
    size = x * P * 4 * density
    if size >= 1024*1024:
        multi_p_ab = GbE_multi_p_ab_large
    else:
        multi_p_ab = GbE_multi_p_ab_small
    a, b = multi_p_ab[P]
    return (a + b * size) * 2

def predict_density_with_size_and_computation(m, comp_time, P):
    alpha = 4*0.436e-3
    beta =  4*9e-6*1e-3
    def _denseallreduce_model(P, m):
        return 2*(P-1)*alpha + 2* (P-1)/P * m * beta

    def _sparseallreduce_model(P, m, rho=0.001):
        return np.log2(P) + 2 * (P - 1) * rho * m * beta

    def _proper_rho_with_sparse_allreduce(P, m, comp_time):
        rho = 0.001
        t = comp_time - np.log2(P) * alpha 
        if t <= 0:
            return rho 
        rho = t/ (2*(P-1)*beta*m)
        if rho > 1.0:
            rho = 0.05
        rho = max(rho, 0.001)
        return rho
    return 0.001
    #if m >= 1024*16:
    #    return 0.001
    #else:
    #    return 1

    #dense_time = _denseallreduce_model(P, m)
    #density = 1
    #if dense_time < comp_time:
    #    return density
    #else:
    #    return _proper_rho_with_sparse_allreduce(P, m, comp_time)

def predict_allreduce_time_with_size(alpha, beta, size, P):
    if size == 0:
        return 0.0
    return alpha + beta * size 

def gen_threshold_from_normal_distribution(p_value, mu, sigma):
    zvalue = stats.norm.ppf((1-p_value)/2)
    return mu+zvalue*sigma, mu-zvalue*sigma

def check_unique(l):
    d = {}
    for k in l:
        if k in d:
            print('element: %s is duplicate in %s' % (k, l))
            return False
        d[k] = 1
    return True

