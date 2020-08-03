import numpy as np
import torch

from statistics.meters import Meter
from .wavelength import hsl_to_rgb
from skimage import color


def deValoisExperiment(model, layer, size=32, stepsize=1, device='cpu', lab=False):
    """
    Experiment for a single cell
    """

    featuremap_position = int(size / 2.0)

    # if plot:
    #     plt.figure()
    #     plt.title(layer + ', ' + str(featuremap_index))
    # stimuli = []
    all_responses = {'hue_responses': {}, 'uniform_responses': {}, 'hues': None}

    all_responses['hues'] = list(range(0, 360, stepsize))

    # for brightness in brightnessModifier:
    responses = []
    hues = []
    for h in all_responses['hues']:
        # energy needs to be normalised: energy of a photon is h*c / wavelength (n=planck's const; c=speed of light)
        # blue has higher energy than red, so we want to attenuate intensity (blue most, red least)
        # energy = 1.0 / wavelength * maxWavelength
        rgb = hsl_to_rgb(h, 1., 0.5)
        #             print(rgb.norm(p=2))
        stimulus = torch.ones((1, size, size, 3)) * torch.Tensor(rgb)

        # if brightness == brightnessModifier[0]:
        #     st = torch.ones((1, 32, 32, 3)) * wavelength_to_rgb(wavelength) / brightness / energy
        #     stimuli.append(st[0].numpy())
        if lab:
            stimulus = torch.from_numpy(color.rgb2lab(stimulus.numpy())).float().div(100)

        stimulus = stimulus.permute(0, 3, 2, 1)

        with torch.no_grad():
            featuremaps = model.forward_to_layer(stimulus.to(device), layer)
            hues.append(h)
            response = featuremaps[0, :, featuremap_position, featuremap_position]
            responses.append(response)
    # if plot:
    #     plt.plot(wls, responses)
    all_responses['hue_responses'] = torch.stack(responses, 0)

    for i in np.arange(0, 1.1, 0.5):
        stimulus = torch.ones((1, size, size, 3)) * i

        if lab:
            stimulus = torch.from_numpy(color.rgb2lab(stimulus.numpy())).float()

        stimulus = stimulus.permute(0, 3, 2, 1)
        featuremaps = model.forward_to_layer(stimulus.to(device), layer)
        r = featuremaps[0, :, featuremap_position, featuremap_position]
        # if plot:
        #     plt.plot(wls, [r] * len(wls))
        all_responses['uniform_responses'][i] = r

    # if plot:
    #     plt.show()
    #     plt.figure()
    #     plt.imshow(np.concatenate(stimuli, axis=1), extent=[wls[0], wls[-1], 0, 16])
    #     plt.show()

    return all_responses


def deValoisExperimentStats(model, layer, spontaneous_level=0, device='cpu', lab=False, size=32):
    """
    Generate classification for a single cell
    """
    # sc = None
    data = deValoisExperiment(model, layer.replace('relu', 'conv'), device=device, lab=lab, size=size)

    spontaneous_rates = data['uniform_responses'][spontaneous_level]

    classes = [''] * spontaneous_rates.size(0)
    max_params = [] * spontaneous_rates.size(0)
    min_params = [] * spontaneous_rates.size(0)
    maxes = [] * spontaneous_rates.size(0)
    mins = [] * spontaneous_rates.size(0)

    responses = data['hue_responses']

    for i in range(len(classes)):
        spontaneous_rate = spontaneous_rates[i]
        response = responses[:, i]

        if torch.all(response == spontaneous_rate):
            classes[i] = 'spectrally unresponsive'
        elif (response.max() > spontaneous_rate) and (response.min() < spontaneous_rate) and (spontaneous_rate > 0.0):
            classes[i] = 'spectrally opponent'
        else:
            classes[i] = 'spectrally non-opponent'

        perm = torch.randperm(response.size(0))
        val, idx = torch.max(response[perm], dim=0)
        max_params.append(data['hues'][perm[idx.item()]])
        maxes.append(val.item())

        perm = torch.randperm(response.size(0))
        val, idx = torch.min(response[perm], dim=0)
        min_params.append(data['hues'][perm[idx.item()]])
        mins.append(val.item())

    return classes, max_params, maxes, min_params, mins, spontaneous_rates


class DeValois(Meter):
    def __init__(self, layers=None, lab=False, size=32):
        if layers is None:
            layers = ['retina_relu2', 'ventral_relu0', 'ventral_relu1']
        super().__init__(['layer', 'cell', 'class', 'max_params', 'max', 'min_params', 'min', 'spontaneous_rate'])
        self.layers = layers
        self.lab = lab
        self.size = size

    def compute(self, model, metadata, device):
        model.eval()
        res = [[], [], [], [], [], [], [], []]
        for layer in self.layers:
            if not model.has_layer(layer):
                continue
            classes, max_params, maxes, min_params, mins, spontaneous_rates =\
                deValoisExperimentStats(model, layer, device=device, lab=self.lab, size=self.size)
            res[0] += [layer] * len(classes)
            res[1] += list(range(len(classes)))
            res[2] += classes
            res[3] += max_params
            res[4] += maxes
            res[5] += min_params
            res[6] += mins
            res[7] += spontaneous_rates.tolist()
        return res
