import torch

def circle_mask(height, width, radius):
    x = torch.linspace(- width / 2 + 0.5, width / 2 - 0.5, width)
    y = torch.linspace(- height / 2 + 0.5, height / 2 - 0.5, height)
    return x[None, :] ** 2 + y[:, None] ** 2 <= radius ** 2