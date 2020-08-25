import torch
import torch.nn as nn

"""Train a single model
"""
import torchvision.transforms as transforms
from torchbearer import Trial, callbacks
from torch import optim
from torch.utils.data import DataLoader
# from imagenet_hdf5 import ImageNetHDF5
from torchvision.datasets import ImageFolder

import argparse
from sklearn.model_selection import ParameterGrid
import pathlib

from model_imagenet import ImageNetModel

parser = argparse.ArgumentParser(description='Intel Training')
parser.add_argument('--arr', default=0, type=int, help='point in job array')
# parser.add_argument('--d-vvs', default=2, type=int, help='ventral depth')
# parser.add_argument('--cache', default=250, type=int, help='cache size')
parser.add_argument('--root', default='../../intel', type=str, help='root')
args = parser.parse_args()

bottlenecks = [1, 2, 4, 8, 16, 32]
ventral_depths = [0, 1, 2, 3, 4]
n_trials = 10

param_grid = ParameterGrid({
    'n_bn': bottlenecks,
    'd_vvs': ventral_depths,
    'rep': list(range(n_trials))
})

params = param_grid[args.arr]
n_bn = params['n_bn']
d_vvs = params['d_vvs']
t = params['rep']

model_file = f'./models/intel/model_{n_bn}_{d_vvs}_{t}.pt'
log_file = f'./logs/intel/model_{n_bn}_{d_vvs}_{t}.csv'

pathlib.Path(model_file).parents[0].mkdir(parents=True, exist_ok=True)
pathlib.Path(log_file).parents[0].mkdir(parents=True, exist_ok=True)

train_transform = transforms.Compose([
    #transforms.Grayscale(),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    # transforms.Resize(128),
    transforms.ToTensor()  # convert to tensor
])
test_transform = transforms.Compose([
    #transforms.Grayscale(),
    transforms.CenterCrop(128),
    # transforms.Resize(128),
    transforms.ToTensor()  # convert to tensor
])

# load data
trainset = ImageFolder(f'{args.root}/seg_train', transform=train_transform)
testset = ImageFolder(f'{args.root}/seg_test', transform=test_transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=15)
testloader = DataLoader(testset, batch_size=256, shuffle=True,  num_workers=15)

model = ImageNetModel(n_bn, d_vvs, n_inch=3, n_classes=6)
# print(model)

optimiser = optim.RMSprop(model.parameters(), alpha=0.9, lr=0.0001, weight_decay=1e-6)
loss_function = nn.CrossEntropyLoss()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy'], callbacks=[callbacks.CSVLogger(log_file)]).to(device)
trial.with_generators(trainloader, test_generator=testloader)
trial.run(epochs=20)

torch.save(model.conv_dict(), model_file)
