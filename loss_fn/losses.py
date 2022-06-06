import math
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.data_utils import get_per_cls_weights

from loss_fn.ot import sinkhorn_loss_joint_IPOT

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, gamma=0., imbalance_beta=0.9999, args=None):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.args = args
        self.gamma = gamma
        self.imbalance_beta = imbalance_beta
        if self.args.imbalance_loss_reweight:
            self.weight = get_per_cls_weights(cls_num_list, imbalance_beta)
        else:
            self.weight = None

    def update(self, **kwargs):
        if self.args.imbalance_loss_reweight:
            if "cls_num_list" in kwargs and kwargs["cls_num_list"] is not None:
                if "imbalance_beta" in kwargs and kwargs["imbalance_beta"] is not None:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], kwargs["imbalance_beta"])
                else:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], self.imbalance_beta)
            else:
                pass
        else:
            logging.info("WARNING: the imbalance weight has not been updated.")
            self.weight = None

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, imbalance_beta=0.9999, args=None):
        super(LDAMLoss, self).__init__()
        self.args = args
        self.imbalance_beta = imbalance_beta
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.max_m = max_m
        assert s > 0
        self.s = s
        if self.args.imbalance_loss_reweight:
            self.weight = get_per_cls_weights(cls_num_list, imbalance_beta)
        else:
            self.weight = None

    def update(self, **kwargs):
        if self.args.imbalance_loss_reweight:
            if "cls_num_list" in kwargs and kwargs["cls_num_list"] is not None:
                if "imbalance_beta" in kwargs and kwargs["imbalance_beta"] is not None:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], kwargs["imbalance_beta"])
                else:
                    self.weight = get_per_cls_weights(kwargs["cls_num_list"], self.imbalance_beta)
            else:
                pass
        else:
            logging.info("WARNING: the imbalance weight has not been updated.")
            self.weight = None

        if "cls_num_list" in kwargs and kwargs["cls_num_list"] is not None:
            cls_num_list = kwargs["cls_num_list"]
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (self.max_m / np.max(m_list))
            # m_list = torch.cuda.FloatTensor(m_list)
            self.m_list = m_list
        else:
            pass

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        self.m_list = torch.cuda.FloatTensor(self.m_list).to(device=x.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


def linear_combination(x, y, epsilon):  
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'): 
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 

class LabelSmoothingCrossEntropy(nn.Module): 
    def __init__(self, epsilon:float=0.1, reduction='mean'): 
        super().__init__() 
        self.epsilon = epsilon 
        self.reduction = reduction 

    def forward(self, preds, target): 
        n = preds.size()[-1] 
        log_preds = F.log_softmax(preds, dim=-1) 
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction) 
        nll = F.nll_loss(log_preds, target, reduction=self.reduction) 
        return linear_combination(loss/n, nll, self.epsilon)







class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast}")
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast.mean()}")
        # logging.info(f"In SupCon, anchor_dot_contrast.device: {anchor_dot_contrast.device}, self.device: {self.device}")


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # logging.info(f"In SupCon, exp_logits.shape: {exp_logits.shape}, exp_logits: {exp_logits.mean()}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # if torch.any(torch.isnan(log_prob)):
        #     log_prob[torch.isnan(log_prob)] = 0.0
        logging.info(f"In SupCon, log_prob.shape: {log_prob.shape}, log_prob: {log_prob.mean()}")

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        # loss[torch.isnan(loss)] = 0.0
        if torch.any(torch.isnan(loss)):
            # loss[torch.isnan(loss)] = 0.0
            logging.info(f"In SupCon, features.shape: {features.shape}, loss: {loss}")
            raise RuntimeError
        loss = loss.view(anchor_count, batch_size).mean()


        return loss


class proxy_align_loss(nn.Module):
    def __init__(self,
            inter_domain_mapping=False,
            inter_domain_class_match=True,
            noise_feat_detach=False,
            noise_contrastive=False,
            inter_domain_mapping_matrix=None,
            inter_domain_weight=0.0, inter_class_weight=1.0,
            noise_supcon_weight=0.1,
            noise_label_shift=10, device=None):
        super(proxy_align_loss, self).__init__()

        self.inter_domain_mapping = inter_domain_mapping
        self.noise_feat_detach = noise_feat_detach
        self.noise_contrastive = noise_contrastive
        self.inter_domain_class_match = inter_domain_class_match
        self.inter_domain_mapping_matrix = inter_domain_mapping_matrix
        self.inter_domain_weight = inter_domain_weight
        self.inter_class_weight = inter_class_weight
        self.noise_supcon_weight = noise_supcon_weight
        self.noise_label_shift = noise_label_shift
        self.device = device
        self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)


    def forward(self, features, labels, real_batch_size):

        time_table = {}
        time_now = time.time()
        # softmax_out = F.softmax(features, dim=1)

        # real_features = softmax_out[:real_batch_size]
        # noise_features = softmax_out[real_batch_size:]

        real_features = features[:real_batch_size]
        noise_features = features[real_batch_size:]

        noise_batch_size = noise_features.shape[0]

        # yonggang
        # if self.feat_detach:
        #     new_features = torch.cat([real_features, noise_features.clone().detach()], dim=0)
        # else:
        #     new_features = features
        # yonggang
        
        if self.inter_domain_mapping:
            noise_features = torch.matmul(noise_features, self.inter_domain_mapping_matrix.to(self.device))
            new_features = torch.cat([real_features, noise_features], dim=0)
        elif self.noise_feat_detach:
            new_features = torch.cat([real_features, noise_features.clone().detach()], dim=0)
        else:
            new_features = features

        # logging.debug(f"real_features.shape: {real_features.shape}, noise_features.shape:{noise_features.shape},\
        #     features.shape: {features.shape}, real_batch_size:{real_batch_size} ")
        # Here the noise_features[:real_batch_size] is designed in order to avoid overflow.

        if real_batch_size > noise_batch_size:
            align_domain_loss = torch.linalg.norm(real_features[:noise_batch_size] - noise_features, ord=2, dim=1).sum() \
                / float(real_batch_size)
        else:
            align_domain_loss = torch.linalg.norm(real_features - noise_features[:real_batch_size], ord=2, dim=1).sum() \
                / float(real_batch_size)
        time_table["align_domain_loss"] = time.time() - time_now
        time_now = time.time()

        # real_labels = labels[:real_batch_size]
        # noise_labels = labels[real_batch_size:] - self.noise_label_shift
        # align_cls_loss = cross_pair_norm(real_labels, real_features, noise_labels, noise_features)

        new_features = F.normalize(new_features, dim=1).unsqueeze(1)
        new_noise_features = F.normalize(noise_features, dim=1).unsqueeze(1)

        if self.inter_domain_class_match:
            real_labels = labels[:real_batch_size]
            noise_labels = labels[real_batch_size:] - self.noise_label_shift
            align_cls_loss = self.supcon_loss(new_features, labels=torch.cat([real_labels, noise_labels], dim=0), temperature=0.07, mask=None)
        else:
            align_cls_loss = self.supcon_loss(new_features, labels=labels, temperature=0.07, mask=None)

        if self.noise_contrastive:
            noise_labels = labels[real_batch_size:] - self.noise_label_shift
            noise_cls_loss = self.supcon_loss(new_noise_features, labels=noise_labels, temperature=0.07, mask=None)
            noise_cls_loss_value = noise_cls_loss.item()
        else:
            noise_cls_loss = 0.0
            noise_cls_loss_value = 0.0

        time_table["align_cls_loss"] = time.time() - time_now
        time_now = time.time()

        # logging.debug(f"Calculating proxy align loss, time: {time_table}")
        return self.inter_domain_weight * align_domain_loss + \
            self.inter_class_weight * align_cls_loss + self.noise_supcon_weight * noise_cls_loss, align_domain_loss.item(), align_cls_loss.item(), noise_cls_loss_value

        # return 0.0 * align_domain_loss + \
        #     self.inter_class_weight * align_cls_loss, align_domain_loss.item(), align_cls_loss.item()




def cross_pair_norm(src_labels, src_features, tgt_labels, tgt_features):
    norm = 0
    count = 0
    for i in range(len(src_labels)):
        for j in range(len(tgt_labels)):
            if src_labels[i] == tgt_labels[j]:
                count += 1
                norm += torch.linalg.norm(src_features[i] - tgt_features[j], ord=2, dim=0).sum()
    return norm / count



def pair_norm(labels, features):
    norm = 0
    count = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                count += 1
                norm += torch.linalg.norm(features[i] - features[j], ord=2, dim=0).sum()
    return norm / count



class align_feature_loss(nn.Module):
    def __init__(self, feature_align_means=None, fed_align_std=None, device=None):
        super(align_feature_loss, self).__init__()
        self.feature_align_means = feature_align_means
        self.fed_align_std = fed_align_std
        if feature_align_means is not None:
            self.num_classes = self.feature_align_means.shape[0]
        self.device = device
        self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)

    def refresh(self, feature_align_means):
        self.feature_align_means = feature_align_means
        self.num_classes = self.feature_align_means.shape[0]

    def forward(self, features, labels, real_batch_size):
        repeat_times = real_batch_size // self.num_classes + 1
        align_labels = torch.tensor(list(range(0, self.num_classes))*repeat_times).to(self.device)
        align_features = self.feature_align_means.repeat(repeat_times, 1).to(self.device)
        align_features = torch.normal(mean=align_features, std=self.fed_align_std)

        all_features = torch.cat([features, align_features], dim=0)

        all_features = F.normalize(all_features, dim=1)
        all_features = all_features.unsqueeze(1)

        align_cls_loss = self.supcon_loss(
            features=all_features,
            labels=torch.cat([labels, align_labels], dim=0),
            temperature=0.07, mask=None)

        return align_cls_loss






class Distance_loss(nn.Module):
    def __init__(self, distance, device=None):
        super(Distance_loss, self).__init__()
        self.distance = distance
        self.device = device
        if self.distance == "SupCon":
            self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)
        else:
            self.supcon_loss = None


    def forward(self, x1, x2, label1=None, label2=None):
        if self.distance == "L2_norm":
            loss = self.L2_norm(x1, x2)
        elif self.distance == "cosine":
            loss = self.cosine(x1, x2)
        elif self.distance == "SupCon":
            loss = self.supcon(x1, x2, label1, label2)
        elif self.distance == "OT":
            loss = self.OT(x1, x2, label1, label2)
        else:
            raise NotImplementedError
        return loss


    def OT(self, feature1, feature2, label1, label2):
        sinkhorn_loss_joint_IPOT(1, 0.00, feature1,
                            feature2, label1, label2,
                            0.01, None, None)

        # sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
        #                     logits_pred, None, None,
        #                     0.01, m, n)

    def L2_norm(self, x1, x2):
        return (x1 - x2).norm(p=2)

    def cosine(self, x1, x2):
        cos = F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos.mean()
        return loss

    def supcon(self, feature1, feature2, label1, label2):

        all_features = torch.cat([feature1, feature2], dim=0)

        all_features = F.normalize(all_features, dim=1)
        all_features = all_features.unsqueeze(1)

        align_cls_loss = self.supcon_loss(
            features=all_features,
            labels=torch.cat([label1, label2], dim=0),
            temperature=0.07, mask=None)
        return align_cls_loss











class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                        for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                    for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss













