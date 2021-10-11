import torch
import numpy as np
from physics.inpainting import Inpainting
from dataset.cvdb import CVDB_ICCV
from models.unet import UNet
from utils.metric import cal_psnr

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Inpainting test.')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--dataset-name', default='Urban100', type=str,
                    help="dataset name=['Urban100'] (default: 'Urban100')."
                         "You can add your test image set under ./dataset/")
parser.add_argument('--sample-to-show', default=[0], nargs='*', type=int,
                    help='the test sample id for visualization'
                         'default [0]')
# specifying path to trained models:
parser.add_argument('--ckp', default='./ckp/inpainting/ckp_ei_final.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of EI net')
parser.add_argument('--model-name', default='EI', type=str, help="name of the trained model (dafault: 'EI')")

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

    psnr_fbp, psnr_net=[],[]

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

        x_hat = test(unet, args.ckp, fbp)

        if i in args.sample_to_show:
            plt.subplot(1,4,1)
            plt.imshow(y.squeeze().detach().permute(1, 2, 0).cpu().numpy())
            plt.title('y')
            
            plt.subplot(1,4,2)
            plt.imshow(fbp.squeeze().detach().permute(1, 2, 0).cpu().numpy())
            plt.title('FBP ({:.2f})'.format(cal_psnr(x, fbp)))
            
            plt.subplot(1,4,3)
            plt.imshow(x_hat.squeeze().detach().permute(1, 2, 0).cpu().numpy())
            plt.title('{} ({:.2f})'.format(args.model_name, cal_psnr(x, fbp)))
            
            plt.subplot(1,4,4)
            plt.imshow(x.squeeze().detach().permute(1, 2, 0).cpu().numpy())
            plt.title('x (GT)')
            
            ax = plt.gca()
            ax.set_xticks([]), ax.set_yticks([])
            plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.9, hspace=0.02, wspace=0.02)
            plt.show()
            
    print('Inpainting (0.3) AVG-PSNR: A^+y={:.2f}\t{}={:.2f}'.format(np.mean(psnr_fbp), args.model_name, np.mean(psnr_ei)))

if __name__=='__main__':
    main()
