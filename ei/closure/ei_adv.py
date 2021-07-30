import torch
import numpy as np
from torch.autograd import Variable
from utils.metric import cal_psnr, cal_mse


def closure_ei_adv(generator, discriminator, dataloader, physics, transform,
                        optimizer_G, optimizer_D, criterion_mc, criterion_ei, criterion_gan,
                        alpha, dtype, device, reportpsnr=False):
    loss_mc_seq, loss_ei_seq, loss_g_seq, loss_G_seq, loss_D_seq, psnr_seq, mse_seq = [], [], [], [], [], [], []

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape)==3:
            x = x.unsqueeze(1)
        x = x.type(dtype).to(device)

        # Measurements
        y0 = physics.A(x)

        # Model range inputs
        x0 = Variable(physics.A_dagger(y0))  # range input (pr)

        # Adversarial ground truths
        valid = torch.ones(x.shape[0], *discriminator.output_shape).type(dtype).to(device)
        valid_ei = torch.ones(x.shape[0]*transform.n_trans, *discriminator.output_shape).type(dtype).to(device)
        fake_ei = torch.zeros(x.shape[0]*transform.n_trans, *discriminator.output_shape).type(dtype).to(device)

        valid = Variable(valid, requires_grad=False)
        valid_ei = Variable(valid_ei, requires_grad=False)
        fake_ei = Variable(fake_ei, requires_grad=False)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images from range input A^+y
        x1 = generator(x0)
        y1 = physics.A(x1)

        # EI: x2, x3
        x2 = transform.apply(x1)
        x3 = generator(physics.A_dagger(physics.A(x2)))

        # Loss measures generator's ability to measurement consistency and ei
        loss_fc = criterion_mc(y1, y0)
        loss_ei = criterion_ei(x3, x2)

        # Loss measures generator's ability to fool the discriminator
        loss_g = criterion_gan(discriminator(x2), valid_ei)

        loss_G = loss_fc + alpha['ei'] * loss_ei + alpha['adv'] * loss_g

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = criterion_gan(discriminator(x1.detach()), valid)
        fake_loss = criterion_gan(discriminator(x2.detach()), fake_ei)
        loss_D = 0.5 * alpha['adv'] * (real_loss + fake_loss)

        loss_D.backward()
        optimizer_D.step()


        if reportpsnr:
            psnr_seq.append(cal_psnr(x1, x))
            mse_seq.append(cal_mse(x1, x))

        # --------------
        #  Log Progress
        # --------------

        loss_mc_seq.append(loss_fc.item())
        loss_ei_seq.append(loss_ei.item())
        loss_g_seq.append(loss_g.item())
        loss_G_seq.append(loss_G.item())# total loss for G
        loss_D_seq.append(loss_D.item())# total loss for D
    #loss: loss_fc, loss_ti, loss_g, loss_G, loss_D

    loss_closure = [np.mean(loss_mc_seq), np.mean(loss_ei_seq), np.mean(loss_g_seq),\
           np.mean(loss_G_seq), np.mean(loss_D_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure
