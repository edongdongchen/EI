import torch
import numpy as np
from utils.metric import cal_psnr

#only for one-shot imaging
def closure_dip(net, dataloader, z, physics,
                    optimizer, criterion_mc,
                    dtype, device, reportpsnr=False):
    loss_dip_seq = []
    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device)

        y0 = physics.A(x.type(dtype).to(device))

        # z = torch.rand_like(x)

        x1 = net(z)
        y1 = physics.A(x1)

        if reportpsnr:
            psnr = cal_psnr(x1, x)
            mse = torch.nn.MSELoss()(x1, x).item()

        loss_mc = criterion_mc(y1, y0)

        loss_dip_seq.append(loss_mc.item())

        optimizer.zero_grad()
        loss_fc.backward()
        optimizer.step()

    loss_closure = [np.mean(loss_dip_seq)]

    if reportpsnr:
        loss_closure.append(psnr)
        loss_closure.append(mse)

    return loss_closure
