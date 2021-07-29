import numpy as np
from utils.metric import cal_psnr, cal_mse

def closure_sup(net, dataloader, physics, optimizer, criterion, dtype, device, reportpsnr=False):
    loss_seq , psnr_seq, mse_seq = [], [], []
    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device)

        y0 = physics.A(x.type(dtype).to(device))
        x0 = physics.A_dagger(y0)  # range input (pr)

        x1 = net(x0)

        loss = criterion(x1, x)

        loss_seq.append(loss.item())

        if reportpsnr:
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss_closure = [np.mean(loss_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure