import torch
import colour


def hsl_to_rgb(h, s, l):
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60.) % 2 - 1))
    m = l - (c / 2.)

    h = h % 360

    if 0 <= h < 60:
        R, G, B = c, x, 0
    elif 60 <= h < 120:
        R, G, B = x, c, 0
    elif 120 <= h < 180:
        R, G, B = 0, c, x
    elif 180 <= h < 240:
        R, G, B = 0, x, c
    elif 240 <= h < 300:
        R, G, B = x, 0, c
    elif 300 <= h < 360:
        R, G, B = c, 0, x

    return [R + m, G + m, B + m]


def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    if not torch.is_tensor(wavelength):
        wavelength = torch.ones(1) * wavelength
    # wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = torch.zeros(1)
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = torch.zeros(1)
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = torch.ones(1)
    elif wavelength >= 490 and wavelength <= 510:
        R = torch.zeros(1)
        G = torch.ones(1)
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = torch.ones(1)
        B = torch.zeros(1)
    elif wavelength >= 580 and wavelength <= 645:
        R = torch.ones(1)
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = torch.zeros(1)
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = torch.zeros(1)
        B = torch.zeros(1)
    else:
        R = torch.zeros(1)
        G = torch.zeros(1)
        B = torch.zeros(1)
    return torch.cat((R, G, B))


def wavelength_to_rgb_2(wavelength):
    XYZ = colour.wavelength_to_XYZ(wavelength)
    RGB = colour.XYZ_to_sRGB(XYZ)
    return torch.Tensor(RGB)