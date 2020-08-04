import torch
from torchvision import transforms
import psychopy.visual
import psychopy.event


def make_grating(freq, theta, phase, sz=32, grating=None, win=None):
    was_none = False
    if win is None:
        was_none = True
        win = psychopy.visual.Window(
            size=(sz, sz),
            units="pix",
            fullscr=False
        )

    if grating is None:
        grating = psychopy.visual.GratingStim(
            win=win,
            units="pix",
            size=(sz * 2, sz * 2)
        )

    grating.phase = phase
    grating.sf = freq / sz
    grating.ori = 90 - theta

    grating.draw()
    win.flip()
    img = win.getMovieFrame()

    if was_none:
        win.close()

    return transforms.ToTensor()(img).unsqueeze_(0)

    # if not torch.is_tensor(theta):
    #     theta = torch.ones(1, requires_grad=False) * theta
    # # omega = [freq * torch.cos(theta), freq * torch.sin(theta)]
    # radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    # [x, y] = torch.meshgrid([torch.linspace(-radius[0], sz[0] - radius[0] - 1, steps=radius[0] * 2), torch.linspace(-radius[1], sz[1] - radius[1] - 1, steps=radius[1] * 2)])
    # stimuli = 0.5 * torch.cos(omega[0] * x + omega[1] * y + phase) + 0.5
    # return stimuli


def make_grating_rg(freq, theta, phase, sz=(32, 32)):
    # this is equi-intensity in the sense that r+g+b=1 everywhere; it's not equiluminant however 
    gr = make_grating(freq, theta, phase, sz=sz)
    im = torch.stack([gr, 1-gr, torch.zeros(sz)], dim=2)
    return im


def make_gaussian(x0, y0, vx, vy, sz=(32, 32)):
    if not torch.is_tensor(x0):
        x0 = torch.ones(1) * x0
    if not torch.is_tensor(y0):
        y0 = torch.ones(1) * y0
    if not torch.is_tensor(vx):
        vx = torch.ones(1) * vx
    if not torch.is_tensor(vy):
        vy = torch.ones(1) * vy

    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = torch.meshgrid([torch.linspace(-radius[0], radius[0], steps=radius[0] * 2), torch.linspace(-radius[1], radius[1], steps=radius[1] * 2)])

    stimuli = (- ((x - x0).pow(2)/(2 * vx) + (y - y0).pow(2)/(2 * vy))).exp()
    return stimuli


def make_gaussian_rg(x0, y0, vx, vy, sz=(32, 32)):
    # this is equi-intensity in the sense that r+g+b=1 everywhere; it's not equiluminant however
    g = make_gaussian(x0, y0, vx, vy, sz=sz)
    im = torch.stack([g, 1-g, torch.zeros(sz)], dim=2)
    return im


def make_dog(x0, y0, vx0, vy0, vx1, vy1, sz=(32, 32)):
    g1 = make_gaussian(x0, y0, vx0, vy0, sz=sz)
    g2 = make_gaussian(x0, y0, vx1, vy1, sz=sz)

    # dog = (g2 - g1).abs()
    im = torch.stack([g2, 1-g1, torch.zeros(sz)], dim=2)
    return im
