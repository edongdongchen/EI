import torch
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float

# --------------------------------
# metric
# --------------------------------
def cal_psnr(a, b):
    # a: prediction
    # b: ground-truth
    alpha = np.sqrt(a.shape[-1] * a.shape[-2])
    return 20*torch.log10(alpha*torch.norm(b, float('inf'))/torch.norm(b-a, 2)).detach().cpu().numpy()

def cal_mse(a, b, mask=None):
    # a: prediction
    # b: ground-truth
    if mask is None:
        return torch.nn.MSELoss()(a, b).item()
    else:
        return torch.nn.MSELoss()(a[mask>0], b[mask>0]).item()

def cal_ssim(a, b, multichannel=False):
    # a: prediction
    # b: ground-truth
    b = img_as_float(b.squeeze().detach().cpu().numpy())
    a = img_as_float(a.squeeze().detach().cpu().numpy())
    return ssim(b, a, data_range=a.max() - a.min(), multichannel=multichannel)
