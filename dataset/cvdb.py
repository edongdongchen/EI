import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset


def CVDB_ICCV(dataset_name='Urban100', mode='train', batch_size=1, shuffle=True, crop_size=(512, 512), resize=True):
    if dataset_name=='Urban100':
        if os.path.exists('../dataset/Urban100/'):
            imgs_path = '../dataset/Urban100/'
        else:
            imgs_path = './dataset/Urban100/'

    if dataset_name in ['091','092','093','094','095','096','097','098','099','100']:
        mode = 'test'
        if os.path.exists('../dataset/Set1/{}/'.format(dataset_name)):
            imgs_path = '../dataset/Set1/{}/'.format(dataset_name)
        else:
            imgs_path = './dataset/Set1/{}/'.format(dataset_name)

    if resize:
        transform_data = transforms.Compose([transforms.CenterCrop(crop_size),
                                             transforms.Resize(256),
                                             transforms.ToTensor()])
    else:
        transform_data = transforms.Compose([transforms.CenterCrop(crop_size),
                                             transforms.ToTensor()])

    if mode == 'train':
        imgs_path = imgs_path + 'train/'
    if mode == 'test':
        imgs_path = imgs_path + 'test/'

    dataset = datasets.ImageFolder(imgs_path, transform=transform_data, target_transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader