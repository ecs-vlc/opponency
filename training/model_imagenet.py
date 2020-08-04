import torch
import torch.nn as nn

from . import BaselineModel


class Flatten(nn.Module):
    """Flatten incoming tensors
    """
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.size()[0], -1)


def init_weights(m):
    """Inplace xavier uniform initialisation with zero bias for conv and linear layers

    :param m: The layer to initialise
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class ImageNetModel(BaselineModel):
    """
    Parameterised implemention of the retina-net - ventral stream architecture.
    Note that the last layer does not have an explicit softmax and will thus output the logits

    :param n_bn: Number of filters in the bottleneck layer
    :param d_vvs: Depth (nuymber of layers) of the ventral part of the model
    :param n_inch: Number of input channels
    """
    def __init__(self, n_bn, d_vvs, n_inch=1):
        super(ImageNetModel, self).__init__(n_bn, d_vvs, n_inch)

        self.retina = []
        self.retina.append(("retina_conv1", nn.Conv2d(n_inch, 32, (9, 9), padding=4)))
        self.retina.append(("retina_relu1", nn.ReLU()))
        self.retina.append(("retina_conv2", nn.Conv2d(32, n_bn, (9, 9), padding=4)))
        self.retina.append(("retina_relu2", nn.ReLU()))

        last_size = n_bn
        self.ventral = []
        for i in range(d_vvs):
            self.ventral.append(("ventral_conv"+str(i), nn.Conv2d(last_size, 32, (9, 9), padding=4)))
            self.ventral.append(("ventral_relu"+str(i), nn.ReLU()))
            last_size = 32

        self.ventral.append(("ventral_pool", nn.AvgPool2d(4, stride=4)))
        self.ventral.append(("ventral_flatten", Flatten()))
        self.ventral.append(("ventral_fc1", nn.Linear(last_size*32*32, 1024)))
        self.ventral.append(("ventral_fc1_relu", nn.ReLU()))
        self.ventral.append(("ventral_fc2", nn.Linear(1024, 1000)))

        for key, module in self.retina:
                self.add_module(key, module)

        for key, module in self.ventral:
                self.add_module(key, module)

        self.apply(init_weights)


if __name__ == '__main__':
    """Train a single model
    """
    import torchvision.transforms as transforms
    from torchbearer import Trial, callbacks
    from torch import optim
    from torch.utils.data import DataLoader
    from imagenet_hdf5 import ImageNetHDF5
    # from torchvision.datasets import ImageNet

    import argparse

    parser = argparse.ArgumentParser(description='Imagenet Training')
    parser.add_argument('--arr', default=0, type=int, help='point in job array')
    parser.add_argument('--d-vvs', default=2, type=int, help='ventral depth')
    parser.add_argument('--cache', default=250, type=int, help='cache size')
    parser.add_argument('--root', type=str, help='root')
    args = parser.parse_args()

    bns = [1, 2, 4, 8, 16, 32]

    n_bn = bns[args.arr % 6]
    rep = args.arr // 6

    model_file = f'./models/imagenet/model_{n_bn}_{args.d_vvs}_{rep}.pt'
    log_file = f'./logs/imagenet/model_{n_bn}_{args.d_vvs}_{rep}.csv'

    train_transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()  # convert to tensor
    ])
    test_transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.CenterCrop(224),
        transforms.Resize(128),
        transforms.ToTensor()  # convert to tensor
    ])

    # load data
    trainset = ImageNetHDF5(f'{args.root}/train', transform=train_transform, cache_size=args.cache)
    testset = ImageNetHDF5(f'{args.root}/val', transform=test_transform, cache_size=args.cache)

    # create data loaders
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=15)
    testloader = DataLoader(testset, batch_size=256, shuffle=True,  num_workers=15)

    model = ImageNetModel(n_bn, args.d_vvs, n_inch=3)
    # print(model)

    optimiser = optim.RMSprop(model.parameters(), alpha=0.9, lr=0.0001, weight_decay=1e-6)
    loss_function = nn.CrossEntropyLoss()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy'], callbacks=[callbacks.CSVLogger(log_file)]).to(device)
    trial.with_generators(trainloader, val_generator=testloader)
    trial.run(epochs=20)

    torch.save(model.conv_dict(), model_file)
