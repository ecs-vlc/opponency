from statistics.meters import Meter
import numpy as np
from statistics.gratings import *
from skimage import color

gratings = []
grating_params = []
for theta in np.arange(0, 2*np.pi, np.pi / 9):
    for phase in np.arange(0, 2*np.pi, np.pi / 16):
        for freq in range(16):
            grating = (make_grating(freq, theta, phase)).unsqueeze(2).repeat(1, 1, 3).unsqueeze(0).permute(0, 3, 2, 1)
            gratings.append(grating)
            grating_params.append({'freq': freq, 'theta': theta, 'phase': phase})
gratings = torch.cat(gratings, dim=0)
gratings.requires_grad = False
gratings_list = gratings.chunk(int(gratings.size(0) / 128.))


@torch.no_grad()
def gratingsExperiment(model, layer, featuremap_position=(16, 16), device='cpu', lab=False, grey=False):
    all_responses = {'grating_responses': {}, 'uniform_responses': {}, 'grating_params': grating_params}

    g_list = gratings_list

    responses = []
    for i in range(len(g_list)):
        stimulus_batch = g_list[i].permute(0, 2, 3, 1)
        if lab:
            stimulus_batch = torch.from_numpy(color.rgb2lab(stimulus_batch.numpy())).float().div(100)
        if grey:
            stimulus_batch = stimulus_batch[:, :, :, 0].unsqueeze(3)
        stimulus_batch = stimulus_batch.permute(0, 3, 1, 2)
        featuremaps = model.forward_to_layer(stimulus_batch.to(device), layer)
        responses.append(featuremaps[:, :, featuremap_position[0], featuremap_position[1]])
    responses = torch.cat(responses, 0)
    all_responses['grating_responses'] = responses

    for i in np.arange(0, 1.1, 0.5):
        stimulus = torch.ones((1, 32, 32, 3)) * i
        if lab:
            stimulus = torch.from_numpy(color.rgb2lab(stimulus.numpy())).float().div(100)
        if grey:
            stimulus = stimulus[:, :, :, 0].unsqueeze(3)
        stimulus = stimulus.permute(0, 3, 1, 2)
        featuremaps = model.forward_to_layer(stimulus.to(device), layer)
        r = featuremaps[0, :, featuremap_position[0], featuremap_position[1]]

        all_responses['uniform_responses'][i] = r

    return all_responses


@torch.no_grad()
def gratingsExperimentStats(model, layer, featuremap_position=(16, 16), spontaneous_level=0, device='cpu', lab=False, grey=False):
    data = gratingsExperiment(model, layer, featuremap_position, device=device, lab=lab, grey=grey)

    spontaneous_rates = data['uniform_responses'][spontaneous_level]
    classes = [''] * spontaneous_rates.size(0)
    max_params = [] * spontaneous_rates.size(0)
    min_params = [] * spontaneous_rates.size(0)
    maxes = [] * spontaneous_rates.size(0)
    mins = [] * spontaneous_rates.size(0)

    responses = data['grating_responses']

    for i in range(len(classes)):
        spontaneous_rate = spontaneous_rates[i]
        response = responses[:, i]

        if torch.all(response == spontaneous_rate):
            classes[i] = 'spatially unresponsive'
        elif torch.all(response <= spontaneous_rate):
            classes[i] = 'spatially non-opponent'  # / inhibitors'
        elif torch.all(response >= spontaneous_rate):
            classes[i] = 'spatially non-opponent'  # / excitators'
        else:
            classes[i] = 'spatially opponent'

        val, idx = torch.max(response, dim=0)
        max_params.append(data['grating_params'][idx.item()])
        maxes.append(val.item())

        val, idx = torch.min(response, dim=0)
        min_params.append(data['grating_params'][idx.item()])
        mins.append(val.item())

    return classes, max_params, maxes, min_params, mins, spontaneous_rates


class SpatialOpponency(Meter):
    def __init__(self, layers=None, lab=False):
        if layers is None:
            layers = ['retina_relu2', 'ventral_relu0', 'ventral_relu1']
        super().__init__(['layer', 'cell', 'class', 'max_params', 'max', 'min_params', 'min', 'spontaneous_rate'])
        self.layers = layers
        self.lab = lab

    def compute(self, model, metadata, device):
        model.eval()
        res = [[], [], [], [], [], [], [], []]
        for layer in self.layers:
            if not model.has_layer(layer):
                continue

            classes, max_params, maxes, min_params, mins, spontaneous_rates = gratingsExperimentStats(model, layer, device=device, lab=self.lab, grey=(metadata['n_ch'] == 1))
            res[0] += [layer] * len(classes)
            res[1] += list(range(len(classes)))
            res[2] += classes
            res[3] += max_params
            res[4] += maxes
            res[5] += min_params
            res[6] += mins
            res[7] += spontaneous_rates.tolist()

        return res
