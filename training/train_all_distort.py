"""
Train models in rotated colour space
"""
from model import BaselineModel
import torchvision.transforms as transforms
from torchbearer import Trial
import torch
import torch.nn as nn
import torchbearer
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pathlib
import numpy as np
from skimage import color

bottlenecks = [1, 2, 4, 8, 16, 32]
ventral_depths = [0, 1, 2, 3, 4]
n_trials = 10
cmode = 'colour-distort'
angle = 90 / 360.

nch = 3


def distort(image):
    image = np.array(image, np.float32, copy=False) / 255.
    image = color.rgb2hsv(image)
    image[:, :, 0] = image[:, :, 0] + angle

    image = color.hsv2rgb(image)
    image = torch.from_numpy(image).float()

    image = image.view(32, 32, 3)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    image = image.transpose(0, 1).transpose(0, 2).contiguous()
    return image.float()


train_transform = transforms.Compose([
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    distort
])
test_transform = transforms.Compose([
    distort
])

# load data
trainset = CIFAR10(".", train=True, download=True, transform=train_transform)
testset = CIFAR10(".", train=False, download=True, transform=test_transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

for n_bn in bottlenecks:
    for d_vvs in ventral_depths:
        for t in range(n_trials):
            model_file = f'./models/{cmode}/model_{n_bn}_{d_vvs}_{t}.pt'
            log_file = f'./logs/{cmode}/model_{n_bn}_{d_vvs}_{t}.csv'

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
            trial.run(epochs=20, verbose=1)
            torch.save(model.conv_dict(), model_file)
