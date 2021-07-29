import os


import torch
from torch.optim import Adam

from models.unet import UNet
from models.discriminator import Discriminator

from .closure.ei import closure_ei
from .closure.ei_adv import closure_ei_adv
from .closure.mc import closure_mc
from .closure.supervised import closure_sup
from .closure.supervised_ei import closure_sup_ei

from utils.nn import adjust_learning_rate
from utils.logger import get_timestamp, LOG

class EI(object):
    def __init__(self, in_channels, out_channels, img_width, img_height, dtype, device):
        super(EI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_width = img_width
        self.img_height = img_height

        self.dtype = dtype
        self.device = device

    def train_ei(self, dataloader, physics, transform, epochs,
                      lr, alpha, ckp_interval, schedule, residual=True,
                      pretrained=None, task='',
                      loss_type='l2', cat=True, report_psnr=False,
                      lr_cos=False):
        save_path = './ckp/{}_ei_{}'.format(get_timestamp(), task)

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=residual,
                         circular_padding=True, cat=cat).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location=self.device)
            generator.load_state_dict(checkpoint['state_dict'])

        if loss_type=='l2':
            criterion_mc = torch.nn.MSELoss().to(self.device)
            criterion_ei = torch.nn.MSELoss().to(self.device)
        if loss_type=='l1':
            criterion_mc = torch.nn.L1Loss().to(self.device)
            criterion_ei = torch.nn.L1Loss().to(self.device)

        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        if report_psnr:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_mc', 'loss_ei', 'loss_total', 'psnr', 'mse'])
        else:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_mc', 'loss_ei', 'loss_total'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)

            loss = closure_ei(generator, dataloader, physics, transform,
                    optimizer, criterion_mc, criterion_ei,
                    alpha, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)
            if report_psnr:
                print('{}\tEpoch[{}/{}]\tmc={:.4e}\tei={:.4e}\tloss={:.4e}\tpsnr={:.4f}\tmse={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))
            else:
                print(
                    '{}\tEpoch[{}/{}]\tmc={:.4e}\tei={:.4e}\tloss={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()



    def train_ei_adv(self, dataloader, physics, transform, epochs,
                  lr, alpha, ckp_interval, schedule, residual=True, pretrained=None, task='',
                         loss_type='l2', cat=True,
                         report_psnr=False, lr_cos=False):
        save_path = './ckp/{}_ei_adv_{}'.format(get_timestamp(), task)

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=residual,
                         circular_padding=True, cat=cat)

        if pretrained:
            checkpoint = torch.load(pretrained)
            generator.load_state_dict(checkpoint['state_dict'])

        discriminator = Discriminator((self.in_channels, self.img_width, self.img_height))

        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)

        if loss_type=='l2':
            criterion_mc = torch.nn.MSELoss().to(self.device)
            criterion_ei = torch.nn.MSELoss().to(self.device)
        if loss_type=='l1':
            criterion_mc = torch.nn.L1Loss().to(self.device)
            criterion_ei = torch.nn.L1Loss().to(self.device)

        criterion_gan = torch.nn.MSELoss().to(self.device)

        optimizer_G = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])
        optimizer_D = Adam(discriminator.parameters(), lr=lr['D'], weight_decay=0)


        if report_psnr:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_mc', 'loss_ei', 'loss_g', 'loss_G', 'loss_D', 'psnr', 'mse'])
        else:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_mc', 'loss_ei', 'loss_g', 'loss_G', 'loss_D'])


        for epoch in range(epochs):
            adjust_learning_rate(optimizer_G, epoch, lr['G'], lr_cos, epochs, schedule)
            adjust_learning_rate(optimizer_D, epoch, lr['D'], lr_cos, epochs, schedule)

            loss = closure_ei_adv(generator, discriminator,
                                       dataloader, physics, transform,
                                       optimizer_G, optimizer_D,
                                       criterion_mc, criterion_ei, criterion_gan,
                                       alpha, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)

            if report_psnr:
                print('{}\tEpoch[{}/{}]\tfc={:.4e}\tti={:.4e}\tg={:.4e}\tG={:.4e}\tD={:.4e}\tpsnr={:.4f}\tmse={:.4e}'
                      .format(get_timestamp(), epoch, epochs, *loss))
            else:
                print('{}\tEpoch[{}/{}]\tfc={:.4e}\tti={:.4e}\tg={:.4e}\tG={:.4e}\tD={:.4e}'
                      .format(get_timestamp(), epoch, epochs, *loss))

            if epoch % ckp_interval == 0 or epoch+1 == epochs:
                state = {'epoch': epoch,
                         'state_dict_G': generator.state_dict(),
                         'state_dict_D': discriminator.state_dict(),
                         'optimizer_G': optimizer_G.state_dict(),
                         'optimizer_D': optimizer_D.state_dict()}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()




    def train_supervised(self, dataloader, physics, epochs, lr, ckp_interval, schedule,
                        residual=True, pretrained=None, task='', loss_type='l2', cat=True,
                        report_psnr=False, lr_cos=False):
        save_path = './ckp/{}_supervised_{}'.format(get_timestamp(), task)

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=residual,
                         circular_padding=True, cat=residual if cat==None else cat).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained)
            generator.load_state_dict(checkpoint['state_dict'])
        if loss_type=='l2':
            criterion = torch.nn.MSELoss().to(self.device)
        if loss_type=='l1':
            criterion = torch.nn.L1Loss().to(self.device)

        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        if report_psnr:
            log = LOG(save_path, filename='training_loss', field_name=['epoch', 'loss', 'psnr', 'mse'])
        else:
            log = LOG(save_path, filename='training_loss', field_name=['epoch', 'loss'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
            loss = closure_sup(generator, dataloader, physics,
                    optimizer, criterion, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)
            if report_psnr:
                print('{}\tEpoch[{}/{}]\tloss={:.4e}\tpsnr={:.4f}\tmse={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))
            else:
                print('{}\tEpoch[{}/{}]\tloss={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()



    def train_supervised_ei(self, dataloader, physics, transform, epochs,
                      lr, alpha, ckp_interval, schedule, residual=True, pretrained=None, task='',
                            loss_type='l2', cat=True, report_psnr=False, lr_cos=False):
        save_path = './ckp/{}_supervised_ei_{}'.format(get_timestamp(), task)

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=residual,
                         circular_padding=True, cat=cat).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained)
            generator.load_state_dict(checkpoint['state_dict'])

        if loss_type=='l2':
            criterion_sup = torch.nn.MSELoss().to(self.device)
            criterion_ei = torch.nn.MSELoss().to(self.device)
        if loss_type=='l1':
            criterion_sup = torch.nn.L1Loss().to(self.device)
            criterion_ei = torch.nn.L1Loss().to(self.device)


        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        if report_psnr:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_sup', 'loss_ei', 'loss_total', 'psnr', 'mse'])
        else:
            log = LOG(save_path, filename='training_loss', field_name=['epoch', 'loss_fc','loss_ti','loss_total'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
            loss = closure_sup_ei(generator, dataloader, physics, transform,
                    optimizer, criterion_sup, criterion_ei,
                    alpha, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)

            if report_psnr:
                print('{}\tEpoch[{}/{}]\t loss_sup={:.4e}\tei={:.4e}\tloss={:.4e}\tpsnr={:.4f}\tmse={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))
            else:
                print('{}\tEpoch[{}/{}]\t loss_sup={:.4e}\tei={:.4e}\tloss={:.4e}'.format(
                    get_timestamp(), epoch, epochs, *loss))


            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()

    def train_mc(self, dataloader, physics, epochs, lr, ckp_interval, schedule,
                     residual=True, pretrained=None, task='',
                     loss_type='l2', cat=True, report_psnr=False, lr_cos=False):
        save_path = './ckp/{}_mc_{}'.format(get_timestamp(), 'res' if residual else '', task)

        os.makedirs(save_path, exist_ok=True)

        generator = UNet(in_channels=self.in_channels,
                         out_channels=self.out_channels,
                         compact=4, residual=residual,
                         circular_padding=True, cat=cat).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained)
            generator.load_state_dict(checkpoint['state_dict'])

        if loss_type=='l2':
            criterion_mc = torch.nn.MSELoss().to(self.device)
        if loss_type=='l1':
            criterion_mc = torch.nn.L1Loss().to(self.device)


        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        if report_psnr:
            log = LOG(save_path, filename='training_loss', field_name=['epoch', 'loss_fc', 'psnr', 'mse'])
        else:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_fc'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)
            loss = closure_mc(generator, dataloader, physics,
                    optimizer, criterion_mc, self.dtype, self.device, report_psnr)

            log.record(epoch + 1, *loss)

            if report_psnr:
                print('{}\tEpoch[{}/{}]\tmc={:.4e}\tpsnr={:.4f}\tmse={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))
            else:
                print('{}\tEpoch[{}/{}]\tmc={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
        log.close()
