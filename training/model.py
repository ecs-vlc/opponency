import torch
import torch.nn as nn
from collections import OrderedDict


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.size()[0], -1)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class BaselineModel(nn.Module):
    """
    Parameterised implemention of the retina-net - ventral stream architecture.
    Note that the last layer does not have an explicit softmax and will thus output the logits
    """
    def __init__(self, n_bn, d_vvs, n_inch=1):
        super(BaselineModel, self).__init__()

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

        self.ventral.append(("ventral_flatten", Flatten()))
        self.ventral.append(("ventral_fc1", nn.Linear(last_size*32*32, 1024)))
        self.ventral.append(("ventral_fc1_relu", nn.ReLU()))
        self.ventral.append(("ventral_fc2", nn.Linear(1024, 10)))

        for key, module in self.retina:
                self.add_module(key, module)

        for key, module in self.ventral:
                self.add_module(key, module)

        self.apply(init_weights)

    def has_layer(self, layer_name):
        for name, module in self.retina:
            if layer_name == name:
                return True
        for name, module in self.ventral:
            if layer_name == name:
                return True
        return False

    def forward_to_layer(self, x, layer_name):
        """
        Forward propagate to a specific named layer and return the result
        """
        for name, module in self.retina:
            x = module(x)
            if layer_name == name:
                return x
        for name, module in self.ventral:
            x = module(x)
            if layer_name == name:
                return x
        return x

    def forward(self, x):
        for name, module in self.retina:
            x = module(x)
        for name, module in self.ventral:
            x = module(x)
        return x


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torchbearer import Trial
    import torchbearer
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10

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

    # load data
    trainset = CIFAR10(".", train=True, download=True, transform=train_transform)
    testset = CIFAR10(".", train=False, download=True, transform=test_transform)

    # create data loaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)

    model = BaselineModel(4, 1)
    print(model)

    optimiser = optim.RMSprop(model.parameters(), alpha=0.9, lr=0.0001, weight_decay=1e-6)
    loss_function = nn.CrossEntropyLoss()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
    trial.with_generators(trainloader, test_generator=testloader)
    trial.run(epochs=20)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)

