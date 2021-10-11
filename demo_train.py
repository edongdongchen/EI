import torch
from torch.utils.data import DataLoader
import argparse

from ei.ei import EI
from physics.ct import CT
from physics.inpainting import Inpainting
from dataset.ctdb import CTData
from dataset.cvdb import CVDB_ICCV
from transforms.rotate import Rotate
from transforms.shift import Shift

parser = argparse.ArgumentParser(description='EI experiment parameters.')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--schedule', nargs='+', type=int,
                    help='learning rate schedule (when to drop lr by 10x),'
                         'default [2000, 3000, 4000] for CT,'
                         'default [500, 1000, 1500] for inpainting')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run '
                         '(default 5000 for CT, 2000 for inpainting)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate '
                                       '(default 5e-4 for CT, 1e-3 for inpainting)',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-8)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 2 for CT, 1 for inpainting)')
parser.add_argument('--ckp-interval', default=500, type=int,
                    help='save checkpoints interval epochs (default: 1000 for CT, 500 for inpainting)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# ei specific configs:
parser.add_argument('--ei-trans', default=3, type=int,
                    help='number of transformations for EI (default: 5 for CT, 3 for inpainting)')
parser.add_argument('--ei-alpha', default=1.0, type=float,
                    help='equivariance strength (default: 100 for CT, 1 for inpainting)')
parser.add_argument('--adv_beta', default=1e-8, type=float,
                    help='adversarial strength (default: 1e-8)')

# inverse problem task configs:
parser.add_argument('--task', default='inpainting', type=str,
                    help="inverse problems=['ct', 'inpainting'] (default: 'inpainting')")
parser.add_argument('--ct-views', default=50, type=int,
                    help='number of radon views for CT task (default: 50)')
parser.add_argument('--mask-rate', default=0.3, type=float,
                    help='mask rate for Inpainting task (default: 0.3)')

def main():
    args = parser.parse_args()

    device=f'cuda:{args.gpu}'
    alpha = {'ei': args.ei_alpha, 'adv': args.adv_beta} # equivariance strength
    lr = {'G': args.lr, 'WD': args.weight_decay}

    assert args.task in ['ct', 'inpainting']
    if args.task=='ct':
        dataloader = DataLoader(dataset=CTData(mode='train'),
                                batch_size=args.batch_size, shuffle=True)
        # forward model A
        physics = CT(img_width=128, radon_view=args.ct_views, circle=False, device=device)
        # transformations group G (used in Equivariant Imaging)
        transform = Rotate(n_trans=args.ei_trans)
        # define Equivariant Imaging model
        ei = EI(in_channels=1, out_channels=1,
                img_width=128, img_height=128,
                dtype=torch.float, device=device)

    if args.task=='inpainting':
        dataloader = CVDB_ICCV(dataset_name='Urban100', mode='train',
                               batch_size=args.batch_size, shuffle=True)
        physics = Inpainting(img_heigth=256, img_width=256, mask_rate=args.mask_rate, device=device)
        transform = Shift(n_trans=args.ei_trans)
        ei = EI(in_channels=3, out_channels=3,
                img_width=256, img_height=256,
                dtype=torch.float, device=device)

    ei.train_ei(dataloader, physics, transform, args.epochs, lr, alpha, args.ckp_interval, args.schedule,
                residual=True, pretrained=args.resume, task=args.task, loss_type='l2',
                cat=True, lr_cos=args.cos, report_psnr=True)


if __name__ == '__main__':
    main()
