import torch
import torch.nn.functional as F

from utils.data_utils import check_device

def cov_non_diag_norm(x, size=[8, 8]):
    # Calculate Covariance Matrix
    cov_sum = 0
    b, c, w, h = x.size()
    # print(f"x.size(): {x.size()}")
    temp_matrix = 1 - torch.eye(int(size[0]*size[1]), dtype=torch.float, requires_grad=False)

    temp_matrix = check_device(temp_matrix, x)
    #     temp_matrix = 1 - torch.eye(22*3, dtype=torch.FloatTensor, require_grad=False)
    down_x = F.interpolate(x, size=size, mode="bilinear")
    # print(f"down_x.shape: {down_x.shape}")
    for i in range(c):
        sub_x = down_x[:, i, :, :]
        # print(f"sub_x.shape: {sub_x.shape}")
        #         down_x = F.interpolate(sub_x, size=size)
        sub_x = sub_x.view(sub_x.size(0), -1)
        # sub_x = sub_x / torch.linalg.norm(sub_x, dim=1, keepdim=True)
        sub_x = sub_x / sub_x.norm(dim=1, keepdim=True)
        cov = torch.mm(sub_x.t(), sub_x)**2 / (b - 1)
        # print(f"cov.shape: {cov.shape}")
        #         cov = torch.sum(torch.bmm(down_x, down_x.transpose(0, 2, 1)), dim=0) / (x.size(0) - 1)
        # print(f"temp_matrix.shape: {temp_matrix.shape}")
        cov_sum += torch.mean(cov * temp_matrix)
    
    return cov_sum

def cov_norm(x, size=[8, 8]):
    # Calculate Covariance Matrix
    cov_sum = 0
    b, c, w, h = x.size()
    # print(f"x.size(): {x.size()}")
    down_x = F.interpolate(x, size=size)
    # print(f"down_x.shape: {down_x.shape}")
    for i in range(c):
        sub_x = down_x[:, i, :, :]
        # print(f"sub_x.shape: {sub_x.shape}")
        sub_x = sub_x.view(sub_x.size(0), -1)
        cov = torch.mm(sub_x.t(), sub_x) / (b - 1)
        # print(f"cov.shape: {cov.shape}")
        cov_sum += torch.mean(cov)
    
    return cov_sum





