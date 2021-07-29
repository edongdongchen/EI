import torch
from ei.ei import EI
from physics.inpainting import Inpainting
from dataset.cvdb import CVDB_ICCV
from transforms.shift import Shift

device='cuda:0'
epochs = 2000
ckp_interval = 500
schedule = [500, 1000, 1500]
alpha = {'ei': 1} # equivariance strength
lr = {'G': 1e-3, 'WD': 1e-8}

dataloader = CVDB_ICCV(dataset_name='Urban100', mode='train', batch_size=1, shuffle=True)

#define inverse problem (forward model), e.g. inpainting task
mask_rate = 0.3
physics = Inpainting(img_heigth=256, img_width=256, mask_rate=mask_rate, device=device)

# define transformation group {T_g}, e.g. random shift
n_trans=3
transform = Shift(n_trans=n_trans)

# define Equivariant Imaging model
ei = EI(in_channels=3, out_channels=3, img_width=256, img_height=256,
        dtype=torch.float, device=device)

ei.train_ei(dataloader, physics, transform, epochs, lr, alpha, ckp_interval,
            schedule, residual=True, pretrained=None, task='inpainting',
            loss_type='l2', cat=True, lr_cos=False, report_psnr=True)