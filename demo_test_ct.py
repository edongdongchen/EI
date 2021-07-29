import torch

from models.unet import UNet
from dataset.ctdb import CTData
from physics.ct import CT
from utils.plot import plot_iccv_img_onerow
from utils.metric import cal_psnr

import argparse

parser = argparse.ArgumentParser(description='Inpainting test.')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--sample-to-show', default=[9], nargs='*', type=int,
                    help='the test sample id for visualization'
                         'default [9]')
parser.add_argument('--ckp_sup', default='./ckp/ct/ckp_sup_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of Supervised net')
parser.add_argument('--ckp_ei', default='./ckp/ct/ckp_ei_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of EI net')
parser.add_argument('--ckp_mc', default='./ckp/ct/ckp_mc_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of Measurement consistent net')

def main():
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'

    unet = UNet(in_channels=1, out_channels=1, compact=4, residual=True,
                circular_padding=True, cat=True).to(device)
    forw = CT(img_width=128, radon_view=50, circle=False, device=device)
    dataloader = torch.utils.data.DataLoader(dataset=CTData(mode='test'),batch_size=1, shuffle=False)

    def test(net, ckp, fbp, adv=False):
        checkpoint = torch.load(ckp, map_location=device)
        net.load_state_dict(checkpoint['state_dict_G' if adv else 'state_dict'])
        net.to(device).eval()
        return net(fbp)

    for i, x in enumerate(dataloader):
        if i in args.sample_to_show:
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            x = x.type(torch.float).to(device)

            y = forw.A(x)

            fbp = forw.A_dagger(y)

            x_hat_mc = test(unet, args.ckp_mc, fbp)
            x_hat_ei = test(unet, args.ckp_ei, fbp)
            x_hat_sup = test(unet, args.ckp_sup, fbp)

            plot_iccv_img_onerow(torch_imgs=[y, fbp, x_hat_mc, x_hat_ei, x_hat_sup, x],
                                 title=['y', 'FBP', 'MC', 'EI', 'Supervised', 'x (GT)'],
                                 text=['', '{:.2f}'.format(cal_psnr(x, fbp)),
                                       '{:.2f}'.format(cal_psnr(x, x_hat_mc)),
                                       '{:.2f}'.format(cal_psnr(x, x_hat_ei)),
                                       '{:.2f}'.format(cal_psnr(x, x_hat_sup)), ''],
                                 text_color='white', xy=[97, 9], fontsize=12,
                                 figsize=(16, 4), save_path=None, show=True)
        else:
            continue

if __name__=='__main__':
    main()