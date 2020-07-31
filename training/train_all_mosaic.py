"""
Train models with input mosaics
"""
import math
from model import BaselineModel
import torchvision.transforms as transforms
import torchvision.transforms.functional as T
from torchbearer import Trial
import torch
import torch.nn as nn
import torchbearer
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pathlib
from sklearn.model_selection import ParameterGrid

import argparse

parser = argparse.ArgumentParser(description='Mosaic')
parser.add_argument('--arr', default=0, type=int, help='point in job array')
args = parser.parse_args()

bottlenecks = [1, 2, 4, 8, 16, 32]
ventral_depths = [0, 1, 2, 3, 4]
n_trials = 10
cmode = 'colour'

param_grid = ParameterGrid({
    'n_bn': bottlenecks,
    'd_vvs': ventral_depths,
    'rep': list(range(n_trials))
})

params = param_grid[args.arr]
n_bn = params['n_bn']
d_vvs = params['d_vvs']
t = params['rep']

nch = 3
train_transform = transforms.Compose([
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor()  # convert to tensor
])
test_transform = transforms.Compose([
    # transforms.ToTensor()  # convert to tensor
])


class Mosaic(torch.utils.data.Dataset):
    """
    Adapted from https://github.com/Cyanogenoid/perm-optim/blob/master/data.py
    """
    def __init__(self, dataset, num_tiles, transform=None, image_size=None):
        self.dataset = dataset
        self.num_tiles = num_tiles
        self.transform = transform if transform is not None else transforms.ToTensor()
        if not image_size:
            image_size = self.dataset[0][0].width
        self.tile_size = self.find_tile_size(image_size)
        self.crop = transforms.CenterCrop(self.tile_size * self.num_tiles)

    def find_tile_size(self, image_size):
        ratio = image_size / self.num_tiles
        return int(math.ceil(ratio))

    def __getitem__(self, item):
        img, label = self.dataset[item]
        # make sure image size is divisible
        if self.num_tiles * self.tile_size != img.width:
            img = T.resize(img, self.tile_size * self.num_tiles)
        if img.width != img.height:
            img = self.crop(img)
        tiles = []
        for i in range(self.num_tiles):
            y = i * self.tile_size
            for j in range(self.num_tiles):
                x = j * self.tile_size
                tile = T.crop(img, x, y, self.tile_size, self.tile_size)
                tile = self.transform(tile)
                # tile = T.to_tensor(tile)
                tiles.append(tile)
        tiles = torch.stack(tiles, dim=0)
        perm = torch.randperm(tiles.size(0))  # sorted to unsorted
        # reverse_perm = torch.zeros(tiles.size(0)).long()  # unsorted to sorted
        # reverse_perm[perm] = torch.arange(tiles.size(0)).long()

        permuted_tiles = tiles[perm].transpose(0, 1)

        img = [[0] * self.num_tiles] * self.num_tiles
        j = 0
        for i in range(self.num_tiles ** 2):
            if not (i == 0) and i % self.num_tiles == 0:
                img[j] = torch.cat(img[j], dim=2)
                j = j + 1
            tile = permuted_tiles[:, i]
            img[j][i % self.num_tiles] = tile
        img[-1] = torch.cat(img[-1], dim=2)
        img = torch.cat(img, dim=1)

        # img = self.transform(T.to_pil_image(img))

        # tiles = tiles.transpose(0, 1)
        return img, label

    def __len__(self):
        return len(self.dataset)


# load data
trainset = Mosaic(CIFAR10(".", train=True, download=True, transform=train_transform), 4)
testset = Mosaic(CIFAR10(".", train=False, download=True, transform=test_transform), 4)

# create data loaders
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

# for n_bn in bottlenecks:
#     for d_vvs in ventral_depths:
#         for t in range(n_trials):
model_file = f'./models/{cmode}-mos/model_{n_bn}_{d_vvs}_{t}.pt'
log_file = f'./logs/{cmode}-mos/model_{n_bn}_{d_vvs}_{t}.csv'

pathlib.Path(model_file).parents[0].mkdir(parents=True, exist_ok=True)
pathlib.Path(log_file).parents[0].mkdir(parents=True, exist_ok=True)

model = BaselineModel(n_bn, d_vvs, nch)

optimiser = optim.RMSprop(model.parameters(), alpha=0.9, lr=0.0001, weight_decay=1e-6)
loss_function = nn.CrossEntropyLoss()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy'],
              callbacks=[
                  # torchbearer.callbacks.imaging.MakeGrid().on_train().to_file('sample.png'),
                  torchbearer.callbacks.csv_logger.CSVLogger(log_file)]).to(device)
trial.with_generators(trainloader, val_generator=testloader)
trial.run(epochs=20)
torch.save(model.conv_dict(), model_file)
