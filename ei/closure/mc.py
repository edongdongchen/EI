import numpy as np
from utils.metric import cal_psnr, cal_mse

def closure_mc(net, dataloader, physics,
                    optimizer, criterion_mc,
                    dtype, device, reportpsnr=False):
    loss_mc_seq, psnr_seq, mse_seq = [], [], []
    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device) # ground-truth

        y0 = physics.A(x.type(dtype).to(device)) # measurement
        x0 = physics.A_dagger(y0)  # range input

        x1 = net(x0)
        y1 = physics.A(x1)

        loss_mc = criterion_mc(y1, y0)

        loss_mc_seq.append(loss_mc.item())

        if reportpsnr:
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss_mc.backward()
        optimizer.step()


    loss_closure = [np.mean(loss_mc_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure