dependencies = ['torch', 'torchvision']

from torch.hub import load_state_dict_from_url


def _baseline_model(mode='colour', pretrained=True, n_bn=32, d_vvs=2, rep=0):
    from training import BaselineModel

    n_inch = 1 if mode == 'grey' else 3
    model = BaselineModel(n_bn, d_vvs, n_inch)

    if pretrained:
        state = load_state_dict_from_url(
            f'http://marc.ecs.soton.ac.uk/pytorch-models/opponency/cifar10/{mode}/model_{n_bn}_{d_vvs}_{rep}.pt',
            progress=True
        )
        try:
            model.load_conv_dict(state)
        except:
            model.load_state_dict(state)

    return model


def _imagenet_model(pretrained=True, n_bn=32, d_vvs=2, rep=0):
    from training import ImageNetModel

    model = ImageNetModel(n_bn, d_vvs, 3)

    if pretrained:
        state = load_state_dict_from_url(
            f'http://marc.ecs.soton.ac.uk/pytorch-models/opponency/imagenet/model_{n_bn}_{d_vvs}_{rep}.pt',
            progress=True
        )
        model.load_conv_dict(state)

    return model


def colour_full(n_bn=32, d_vvs=2, rep=0):
    return _baseline_model(mode='colour_full', n_bn=n_bn, d_vvs=d_vvs, rep=rep)


def colour(n_bn=32, d_vvs=2, rep=0):
    return _baseline_model(mode='colour_full', n_bn=n_bn, d_vvs=d_vvs, rep=rep)


def shuffled(n_bn=32, d_vvs=2, rep=0):
    return _baseline_model(mode='shuffled', n_bn=n_bn, d_vvs=d_vvs, rep=rep)


def mosaic(n_bn=32, d_vvs=2, rep=0):
    return _baseline_model(mode='mosaic', n_bn=n_bn, d_vvs=d_vvs, rep=rep)


def cielab(n_bn=32, d_vvs=2, rep=0):
    return _baseline_model(mode='cielab', n_bn=n_bn, d_vvs=d_vvs, rep=rep)


def distorted(n_bn=32, d_vvs=2, rep=0):
    return _baseline_model(mode='mosaic', n_bn=n_bn, d_vvs=d_vvs, rep=rep)


def grey(n_bn=32, d_vvs=2, rep=0):
    return _baseline_model(mode='grey', n_bn=n_bn, d_vvs=d_vvs, rep=rep)


def imagenet(n_bn=32, d_vvs=2, rep=0):
    return _imagenet_model(n_bn=n_bn, d_vvs=d_vvs, rep=rep)
