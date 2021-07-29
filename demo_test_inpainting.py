import torch
import numpy as np
from physics.inpainting import Inpainting
from dataset.cvdb import CVDB_ICCV
from models.unet import UNet
from utils.metric import cal_psnr
from utils.plot import plot_iccv_img_onerow

import argparse

parser = argparse.ArgumentParser(description='Inpainting test.')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--dataset-name', default='Urban100', type=str,
                    help="dataset name=['Urban100'] (default: 'Urban100')."
                         "You can add your test image set under ./dataset/")
parser.add_argument('--sample-to-show', default=[0], nargs='*', type=int,
                    help='the test sample id for visualization'
                         'default [0]')
# specifying path to trained models:
parser.add_argument('--ckp_sup', default='./ckp/inpainting/ckp_sup_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of Supervised net')
parser.add_argument('--ckp_ei', default='./ckp/inpainting/ckp_ei_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of EI net')
parser.add_argument('--ckp_mc', default='./ckp/inpainting/ckp_mc_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of Measurement consistent net')
parser.add_argument('--ckp_ei_sup', default='./ckp/inpainting/ckp_sup_ei_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of EI regularized supervised net')
parser.add_argument('--ckp_ei_adv', default='./ckp/inpainting/ckp_ei_adv_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of Adversarial EI net')
def main():
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'

    # define the dataloader (i.e. 'urban100', first 90 imgs for training, last 10 for testing)
    dataloader = CVDB_ICCV(dataset_name=args.dataset_name, mode='test',
                           batch_size=1, shuffle=False)

    # define the forward oeprator (i.e. physics)
    forw = Inpainting(img_heigth=256, img_width=256, mask_rate=0.3, device=device)

    # define the network G (i.e. residual unet in the paper)
    unet = UNet(in_channels=3, out_channels=3, compact=4, residual=True,
                circular_padding=True, cat=True).to(device)

    psnr_fbp, psnr_ei, psnr_mc, psnr_sup, psnr_ei_sup, psnr_ei_adv=[],[],[],[],[],[]

    def test(net, ckp, fbp, adv=False):
        checkpoint = torch.load(ckp, map_location=device)
        net.load_state_dict(checkpoint['state_dict_G' if adv else 'state_dict'])
        net.to(device).eval()
        return net(fbp)

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # groundtruth
        x = x.type(torch.float).to(device)
        # compute measurement
        y = forw.A(x)
        # compute the A^+y or FBP
        fbp = forw.A_dagger(y)

        x_hat_mc = test(unet, args.ckp_mc, fbp)
        x_hat_ei = test(unet, args.ckp_ei, fbp)
        x_hat_sup = test(unet, args.ckp_sup, fbp)
        x_hat_ei_sup = test(unet, args.ckp_ei_sup, fbp)
        x_hat_ei_adv = test(unet, args.ckp_ei_adv, fbp, adv=True)

        psnr_fbp.append(cal_psnr(fbp, x))
        psnr_ei.append(cal_psnr(x_hat_ei, x))
        psnr_mc.append(cal_psnr(x_hat_mc, x))
        psnr_sup.append(cal_psnr(x_hat_sup, x))
        psnr_ei_sup.append(cal_psnr(x_hat_ei_sup, x))
        psnr_ei_adv.append(cal_psnr(x_hat_ei_adv, x))

        if i in args.sample_to_show:
            plot_iccv_img_onerow(torch_imgs=[y, fbp, x_hat_mc, x_hat_ei, x_hat_sup, x],
                                 title=['y', r'$A^\dagger y$', 'MC', 'EI', 'Supervised', 'x (GT)'],
                                 text=['', '{:.2f}'.format(cal_psnr(fbp, x)),
                                       '{:.2f}'.format(cal_psnr(x_hat_mc, x)),
                                       '{:.2f}'.format(cal_psnr(x_hat_ei, x)),
                                       '{:.2f}'.format(cal_psnr(x_hat_sup, x)), ''],
                                 text_color='yellow',
                                 fontsize= 12, xy=[190, 20],
                                 figsize=(16, 4), save_path=None, show=True)

    print('Inpainting (0.3) AVG-PSNR: A^+y={:.2f}\tMC={:.2f}\tEI={:.2f}\tEI_adv={:.2f}\tSup={:.2f}\tEI_sup={:.2f}'.format(
        np.mean(psnr_fbp), np.mean(psnr_mc), np.mean(psnr_ei),
        np.mean(psnr_ei_adv), np.mean(psnr_sup), np.mean(psnr_ei_sup)))

if __name__=='__main__':
    main()