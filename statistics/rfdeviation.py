import torch

from statistics.meters import Meter


def deprocess_image(x):
    x -= x.mean()
    if x.std() > 1e-5:
        x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5

    if x.shape[0] == 1:
        x = torch.cat((x, x, x))

    # convert to RGB array
    x = x.mul(255).clamp(0, 255).permute(1, 2, 0)
    return x


class RFDeviation(Meter):
    """This Meter computes the standard deviation of the RF for a selection of random inputs. This approximates a
    measure of the linearity of the average feature for the layer. Lower value = more simple cells, higher value = more
    complex cells"""
    def __init__(self, layers=None, repeats=5):
        if layers is None:
            layers = ['retina_conv2', 'ventral_conv0', 'ventral_conv1']
        super().__init__(['layer', 'rfdeviation'])
        self.layers = layers
        self.repeats = repeats

    def compute(self, model, metadata, device):
        model.eval()
        res = [[], []]
        for layer in self.layers:
            if not model.has_layer(layer):
                continue
            filters = 32 if layer is not 'retina_conv2' else metadata['n_bn']
            stds = []
            for i in range(filters):
                images = []
                for j in range(self.repeats):
                    input_img = torch.rand(1, metadata['n_ch'], 32, 32).to(device)
                    input_img.requires_grad = True
                    featuremaps = model.forward_to_layer(input_img, layer)
                    loss = torch.mean(featuremaps[0, i, 16, 16])
                    loss.backward()
                    grad = input_img.grad[0].detach()
                    grad = grad / (grad.pow(2).mean().sqrt() + 1e-5)
                    images.append(deprocess_image(grad))
                images = torch.stack(images, dim=0)
                stds.append(images.std(dim=0).mean().item())
            res[0] += [layer] * filters
            res[1] += stds
        return res
