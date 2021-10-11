import torch

from models.unet import UNet
from dataset.ctdb import CTData
from physics.ct import CT
from utils.metric import cal_psnr

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Inpainting test.')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--sample-to-show', default=[9], nargs='*', type=int,
                    help='the test sample id for visualization'
                         'default [9]')
parser.add_argument('--ckp', default='./ckp/ct/ckp_ei_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of a trained model')
parser.add_argument('--model-name', default='EI', type=str, help="name of the trained model (dafault: 'EI')")

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
            x_hat = test(unet, args.ckp_net, fbp)

            plt.subplot(1,4,1)
            plt.imshow(y[0].detach().permute(1, 2, 0).cpu().numpy())
            plt.title('y')
            
            plt.subplot(1,4,2)
            plt.imshow(fbp[0].detach().permute(1, 2, 0).cpu().numpy())
            plt.title('FBP ({:.2f})'.format(cal_psnr(x, fbp)))
            
            plt.subplot(1,4,3)
            plt.imshow(x_hat[0].detach().permute(1, 2, 0).cpu().numpy())
            plt.title('{} ({:.2f})'.format(args.model_name, cal_psnr(x, x_hat)))
            
            plt.subplot(1,4,4)
            plt.imshow(x[0].detach().permute(1, 2, 0).cpu().numpy())
            plt.title('x (GT)')
            
            ax = plt.gca()
            ax.set_xticks([]), ax.set_yticks([])
            plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.9, hspace=0.02, wspace=0.02)
            plt.show()
        else:
            continue
if __name__=='__main__':
    main()
