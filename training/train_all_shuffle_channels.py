from training.model import BaselineModel
import torchvision.transforms as transforms
from torchbearer import Trial
import torch
import torch.nn as nn
import torchbearer
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pathlib

bottlenecks = [1, 2, 4, 8, 16, 32]
ventral_depths = [0, 1, 2, 3, 4]
n_trials = 10
cmode='colour'

if cmode == 'grey':
    nch = 1
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()  # convert to tensor
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()  # convert to tensor
    ])
else:
    nch = 3
    train_transform = transforms.Compose([
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()  # convert to tensor
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()  # convert to tensor
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
            model_file = f'./models/{cmode}-ch/model_{n_bn}_{d_vvs}_{t}.pt'
            log_file = f'./logs/{cmode}-ch/model_{n_bn}_{d_vvs}_{t}.csv'

            pathlib.Path(model_file).parents[0].mkdir(parents=True, exist_ok=True)
            pathlib.Path(log_file).parents[0].mkdir(parents=True, exist_ok=True)

            model = BaselineModel(n_bn, d_vvs, nch)

            optimiser = optim.RMSprop(model.parameters(), alpha=0.9, lr=0.0001, weight_decay=1e-6)
            loss_function = nn.CrossEntropyLoss()

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            @torchbearer.callbacks.on_sample
            def shuffle(state):
                img = state[torchbearer.X]
                x = img.permute(0, 2, 3, 1)
                # x = torch.stack([x[i, :, :, torch.randperm(x.size(3))] for i in range(x.size(0))], dim=0)
                x = x[:, :, :, torch.randperm(x.size(3))]
                state[torchbearer.X] = x.view(img.size(0), img.size(2), img.size(3), img.size(1)).permute(0, 3, 1, 2)

            trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy'],
                          callbacks=[shuffle, torchbearer.callbacks.csv_logger.CSVLogger(log_file)]).to(device)
            trial.with_generators(trainloader, val_generator=testloader)
            trial.run(epochs=20)
            torch.save(model.state_dict(), model_file)
